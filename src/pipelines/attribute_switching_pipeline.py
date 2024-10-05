import torch
import numpy as np
import random
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.fairness import extract_and_classify_nouns, insert_classes_before_noun, check_existing_attribute

class AttributeSwitchingPipeline(StableDiffusionPipeline):
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
        self.attribute_switch_steps = {}
        self.attribute_default_assignments = {}
        self.attribute_weights = {}
        self.use_debiasing = True
        self.selected_nouns = {}
        

    def set_attribute_params(self, attribute: str, switch_step: int, weights: List[int], default_assignments: List[int] = None):
        self.attribute_default_assignments[attribute] = default_assignments # should have size [batch_size] with index of desired class for attribute
        self.attribute_weights[attribute] = weights
        self.attribute_switch_steps[attribute] = switch_step

    def set_debiasing_options(self, use_debiasing: bool):
        self.use_debiasing = use_debiasing

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
        prompts: List[str], 
        prompt_embeds: torch.Tensor,
        ) -> Tuple[List[str], torch.Tensor]:
        
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
        assignments = None
        if default_assignments:
            # If default assignments are given, we always apply them
            assignments = default_assignments
        else:
            # If we do not use distribution guidance or default assignments, we choose them randomly
            assignments = random.choices(range(len(self.get_attribute_classes(attribute))), weights=self.attribute_weights[attribute], k=len(prompts))
        
        debiased_prompts = []
        for i, (prompt, assignment) in enumerate(zip(prompts_to_debias, assignments)):
            if assignment == -1:
                debiased_prompts.append(prompt)
                continue
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

        
        return debiased_prompts, new_prompt_embeds


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
        self.selected_nouns = {}
        
        latents_list = []
        h_hvects_list = []
        #print(f"Debiasing is active, tau_bias: {self.tau_bias}" if self.use_debiasing else "Debiasing not activated!")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Check if we need to perform debiasing for any attribute at this step
                if self.use_debiasing:
                    for attr in self.attribute_switch_steps.keys():
                        if i == self.attribute_switch_steps[attr]:
                            #print(f"Debiasing attribute {attr}...")
                            prompt, prompt_embeds = self.debias_attribute(attr, prompt, prompt_embeds)
                
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
                #latents_list.append(noise_pred.chunk(2)[0].detach().clone())
                h_vects = results[1]
                h_hvects_list.append(h_vects)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                latents_list.append(latents.detach().clone())
                
                # call the callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
                            
                # update progress bar
                progress_bar.update()

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        #image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        has_nsfw_concept = False

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, latents_list, h_hvects_list)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)      
    
      
  
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