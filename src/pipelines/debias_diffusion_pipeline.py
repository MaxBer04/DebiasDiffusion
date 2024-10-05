import torch
import itertools
import numpy as np
import random
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from scipy.interpolate import interp1d
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.fairness import extract_and_classify_nouns, insert_classes_before_noun, check_existing_attribute
from utils.optimal_transport import solve_optimal_transport
from utils.classifier import make_classifier_model
from utils.custom_unet import CustomUNet2DConditionModel

class DebiasDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = False,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)
        self.attribute_classifiers = {}
        self.attribute_distributions = {}
        self.attribute_switch_steps = {}
        self.attribute_bias_ranges = {}
        self.iota_step_range = [4,19] 
        self.tau_bias = 19
        self.attribute_default_assignments = {}
        self.attribute_default_switch_step = {}
        self.use_debiasing = True
        self.use_distribution_guidance = True
        self.selected_nouns = {}
        self.interpolation_method = 'linear'
        self.collect_probs = False
        #self.include_entities = include_entities
        
        # Replace the original UNet with our custom UNet
        custom_unet = CustomUNet2DConditionModel.from_config(self.unet.config)
        custom_unet.load_state_dict(self.unet.state_dict())
        self.unet = custom_unet
        
        
    def set_tau_bias(self, step: int):
      self.tau_bias = step
      
    def set_iota_step_range(self, iota_step_range: List[int]):
        self.iota_step_range = iota_step_range

    def set_attribute_params(self, attribute: str, distribution: List[float], bias_range: Tuple[float, float], 
                            classifier_path: str, num_classes: int, model_type: str, default_assignments: torch.Tensor,
                            default_switch_step: int):
        self.attribute_distributions[attribute] = distribution
        self.attribute_bias_ranges[attribute] = bias_range
        self.attribute_default_assignments[attribute] = default_assignments # should have size [batch_size] with index of desired class for attribute
        self.attribute_default_switch_step[attribute] = default_switch_step
        
        classifier = make_classifier_model(
            in_channels=1280,
            image_size=8,
            out_channels=num_classes,
            model_type=model_type
        )
        state_dict = torch.load(classifier_path, map_location=self.device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()} # fix
        classifier.load_state_dict(new_state_dict)
        
        if self.device.type == 'cuda':
            classifier = classifier.half().to(self.device)
        else:
            classifier = classifier.float().to(self.device)
        
        self.attribute_classifiers[attribute] = classifier

    def set_debiasing_options(self, use_debiasing: bool, use_distribution_guidance: bool, interpolation_method: str = 'linear'):
        self.use_debiasing = use_debiasing
        self.use_distribution_guidance = use_distribution_guidance
        self.interpolation_method = interpolation_method

    def compute_probs(self, h_vects, classifier, t):
        with torch.no_grad():
            logits = classifier(h_vects, t)
        probs = torch.softmax(logits, dim=1).cpu()
        return probs

    def bias_metric(self, p_tar, h_vects, classifier, t):
        
        if isinstance(p_tar, torch.Tensor):
            p_tar = p_tar.cpu().numpy()
        elif isinstance(p_tar, list):
            p_tar = np.array(p_tar)

        probs = self.compute_probs(h_vects, classifier, t)
        probs = np.array(probs)
        expected_distribution = np.mean(probs, axis=0)
        diff = expected_distribution - p_tar
        metric_value = np.linalg.norm(diff) ** 2
        
        return metric_value

    def interpolate_switch_step(self, bias_value, bias_range, timestep_range):
        min_bias, max_bias = bias_range
        min_step, max_step = timestep_range
        
        if self.interpolation_method == 'linear':
            interpolator = interp1d([min_bias, max_bias], [max_step, min_step], bounds_error=False, fill_value=(max_step, min_step))
        elif self.interpolation_method == 'cosine':
            x = np.array([min_bias, min_bias])
            y = np.array([max_step, max_step])
            interpolator = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(max_step, min_step))
        
        return int(interpolator(bias_value))

    def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
        ):
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            if prompt_embeds is None:
                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                    )
                    print(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                )
                prompt_embeds = prompt_embeds[0]

            if self.text_encoder is not None:
                prompt_embeds_dtype = self.text_encoder.dtype
            elif self.unet is not None:
                prompt_embeds_dtype = self.unet.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                max_length = prompt_embeds.shape[1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = uncond_input.attention_mask.to(device)
                else:
                    attention_mask = None

                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            return prompt_embeds, negative_prompt_embeds

    def get_attribute_classes(self, attribute):
        attribute_classes = {
            'gender': ['male', 'female'],
            'age': ['young', 'old'],
            'race': ['white', 'black', 'asian', 'indian']
        }
        return attribute_classes[attribute]
    
    def debias_attribute(
        self, 
        attribute: str, 
        h_vectors: torch.Tensor, 
        timestep: int, 
        prompts: List[str], 
        prompt_embeds: torch.Tensor, 
        default_assignments: torch.Tensor = None,
        ) -> Tuple[List[str], torch.Tensor]:
        
        classifier = self.attribute_classifiers[attribute]
        target_distribution = self.attribute_distributions[attribute] # desired distributions for each attribute 
        
        prompts_to_debias = []
        needs_update = False
        
        # This is the Syntactic Filtering
        # Debias each prompt independently
        for i, prompt in enumerate(prompts):
            # Extract nouns and modifiers, if not done already, and classify if they are human (is_human)
            if i not in self.selected_nouns:
                nouns = extract_and_classify_nouns(prompt)
                human_nouns = [n for n in nouns if n[3]]  # n[3] ist der is_human Boolean
                
                if not human_nouns:
                    self.selected_nouns[i] = None
                else:
                    # If human related nouns are detected, randomly select one
                    self.selected_nouns[i] = random.choice(human_nouns)
            
            selected_noun = self.selected_nouns[i]
            
            # Continue with normal prompt if no human related noun is detected
            if selected_noun is None:
                prompts_to_debias.append(prompt)
                continue
            
            
            chosen_noun, idx, token, _ = selected_noun
            
            # Checks whether a class for the current attribute was already specified for the noun
            # (f.e. 'A female police officer', female for gender would already be defined).
            # Continue with the normal prompt.
            if check_existing_attribute(prompt, chosen_noun, self.get_attribute_classes(attribute)):
                prompts_to_debias.append(prompt)
                continue
            
            needs_update = True
            prompts_to_debias.append(prompt) 
        
        if not needs_update:
            return prompts, prompt_embeds, []
        
        # This is the Distribution Guidance/Sample Guidance
        # If distribution guidance and default_assignments are given (which should not be), default_assigments have priority
        default_assignments = self.attribute_default_assignments[attribute]
        if self.use_distribution_guidance and not default_assignments:
            # Checks are over, prompt has to be debiased for this attribute
            # From here on, it is a matter of which class to select for the given attribute
            # (and consequently insert before the noun into the prompt)
            with torch.no_grad():
                logits = classifier(h_vectors, int(timestep)) 
            probs = torch.softmax(logits, dim=1) # The attributes predictions for the given timestep from our trained h-classifier.
            target_samples = np.random.choice(len(target_distribution), size=h_vectors.shape[0], p=target_distribution) # Sample target vectors from desired target distribution of the attribute
            
            # Solve optimal transport for all batch elements and sampled vectors
            # F.e. if we have gender female/male with target distr. 50%/50% and 2 images. We have probs [.7, .3] (female, male) for image one
            # and probs [.2, .8] for sample two. Then it would be much better if we assign target class female to sample one and male to sample two than vice versa
            # since they are already "closer" in terms of target distribution and predicted probabilites.
            assignments = solve_optimal_transport(probs.cpu().numpy(), target_samples) 
        elif default_assignments:
            # If default assignments are given, we always apply them
            probs = torch.zeros(h_vectors.shape[0], len(self.attribute_distributions[attribute]), device=h_vectors.device)
            assignments = default_assignments
            probs[torch.arange(h_vectors.shape[0]), torch.tensor(default_assignments)] = 1
        else:
            # If we do not use distribution guidance or default assignments, we choose the desired classes for each sample randomly. 
            probs = torch.zeros(h_vectors.shape[0], len(self.attribute_distributions[attribute]), device=h_vectors.device)
            assignments = np.random.choice(len(target_distribution), size=h_vectors.shape[0], p=target_distribution)
            probs[torch.arange(h_vectors.shape[0]), torch.tensor(assignments)] = 1
        
        debiased_prompts = []
        for i, (prompt, assignment) in enumerate(zip(prompts_to_debias, assignments)):
            selected_noun = self.selected_nouns[i]
            
            # If we have a noun to debias then we use the computed assignments (i.e. the desired classes) per prompt or sample.
            # We insert the desired class label into the prompt before the chosen noun and get back a debiased prompt
            # (f.e. from 'A photo of a police officer' to 'A photo of a female police officer')
            if selected_noun:
                chosen_noun, idx, token, _ = selected_noun
                attribute_class = self.get_attribute_classes(attribute)[assignment]
                debiased_prompts.append(insert_classes_before_noun(prompt, chosen_noun, idx, token, [attribute_class]))
            else:
                debiased_prompts.append(prompt)
        
        #print(f"Original prompts: {prompts}")
        #print(f"Debiased prompts: {debiased_prompts}")
        
        # This is the "Actual Debiasing"
        # We reembed the new debiased prompt and return the new prompt as well as the new embedding.
        # Therefore, we incentivize the model to generate the desired class and also provide the new prompt for subsequent
        # debisaing steps with other attributes.
        new_prompt_embeds, new_negative_prompt_embeds = self.encode_prompt(
            debiased_prompts,
            self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        new_prompt_embeds = torch.cat([new_negative_prompt_embeds, new_prompt_embeds])

        
        return debiased_prompts, new_prompt_embeds, probs


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = num_images_per_prompt
            prompt = [prompt]*num_images_per_prompt
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            num_images_per_prompt = 1
        else:
            batch_size = prompt_embeds.shape[0]
            
        negative_prompt = [""]*batch_size

        device = self._execution_device
        self.do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, device, 1, self.do_classifier_free_guidance, negative_prompt,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size,
            self.unet.config.in_channels,
            height, width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        all_probs = {attr: [] for attr in self.attribute_classifiers.keys()}
        self.selected_nouns = {}
        
        latents_list = []
        
        #print(f"Debiasing is active, tau_bias: {self.tau_bias}" if self.use_debiasing else "Debiasing not activated!")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                results = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
                
                noise_pred = results[0]
                h_vects = results[1]
                
                if i <= self.tau_bias and not self.attribute_switch_steps.keys():
                    latents_list.append(latents)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

                if self.collect_probs and i < num_inference_steps:
                    for attr in self.attribute_classifiers.keys():
                        classifier = self.attribute_classifiers[attr]
                        all_probs[attr].append(np.array(self.compute_probs(h_vects, classifier, i)))

                # Check if we need to perform debiasing for any attribute at this step
                if self.use_debiasing:
                    if i == self.tau_bias:
                        for attr in self.attribute_classifiers.keys():
                            
                            # Check if switch steps were already computed (for cases where generation is reset)
                            if attr in self.attribute_switch_steps.keys():
                                continue
                            
                            # Use default switch_steps if provided
                            # Else compute using the bias values
                            if self.attribute_default_switch_step[attr]:
                                self.attribute_switch_steps[attr] = self.attribute_default_switch_step[attr]
                            else:
                                # Compute switching timesteps \iota_i for each attribute
                                classifier = self.attribute_classifiers[attr]
                                bias_value = self.bias_metric(self.attribute_distributions[attr], h_vects, classifier, i)
                                self.attribute_switch_steps[attr] = self.interpolate_switch_step(
                                    bias_value, 
                                    self.attribute_bias_ranges[attr],
                                    self.iota_step_range,
                                )
                                print(f"{attr}: {bias_value}")
                                #('-'*80)
                                #print(f"Attribute {attr}: {bias_value} | {self.attribute_switch_steps[attr]}")
                                
                        test = [self.attribute_switch_steps[attr] for attr in self.attribute_switch_steps.keys()]
                        min_step = np.min(test)
                        if min_step < i:
                            total_steps = num_inference_steps - min_step
                            steps = range(min_step, num_inference_steps+1)
                            timesteps_remaining = timesteps[min_step:]
                            latents = latents_list[min_step]
                            latents = self.continue_generation_early(latents, prompt, prompt_embeds, total_steps, steps, timesteps_remaining, guidance_scale, guidance_rescale, extra_step_kwargs, callback, callback_steps)
                            break
                                
                    # Check if any attribute needs debiasing at this timestep
                    for attr in self.attribute_switch_steps.keys():
                        if i == self.attribute_switch_steps[attr]:
                            #print(f"Debiasing attribute {attr}...")
                            prompt, prompt_embeds, _ = self.debias_attribute(attr, h_vects, i, prompt, prompt_embeds)
                            
                # update progress bar
                progress_bar.update()

        self.attribute_switch_steps = {}

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, h_vects, all_probs) #, all_probs

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
    def continue_generation_early(self, latents, prompt, prompt_embeds, total_steps, steps, timesteps, guidance_scale, guidance_rescale, extra_step_kwargs, callback, callback_steps):
        with self.progress_bar(total=total_steps) as progress_bar:
            for i, t in zip(steps, timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                results = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
                
                noise_pred = results[0]
                h_vects = results[1]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

                # Check if any attribute needs debiasing at this timestep
                for attr in self.attribute_switch_steps.keys():
                    if i == self.attribute_switch_steps[attr]:
                        #print(f"Debiasing attribute {attr}...")
                        prompt, prompt_embeds, _ = self.debias_attribute(attr, h_vects, i, prompt, prompt_embeds)
                            
                # update progress bar
                progress_bar.update()
        return latents       
    
      
  
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg