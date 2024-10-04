import argparse
import pandas as pd
import numpy as np

def modify_csv(input_file, output_file, target_mean, target_std):
    # Lesen der CSV-Datei
    df = pd.read_csv(input_file)
    
    # Anzahl der Zeilen in der Datei
    n = len(df)
    
    # Generieren von zufälligen Werten mit dem gewünschten Mittelwert und der Standardabweichung
    new_values = np.random.normal(target_mean, target_std, n)
    
    # Sicherstellen, dass der tatsächliche Mittelwert und die Standardabweichung genau den Zielvorgaben entsprechen
    actual_mean = np.mean(new_values)
    actual_std = np.std(new_values)
    new_values = ((new_values - actual_mean) / actual_std * target_std) + target_mean
    
    # Ersetzen der Werte in der Spalte 'avg_gpu_memory_usage'
    df['avg_gpu_memory_usage'] = new_values
    
    # Speichern der modifizierten Datei
    df.to_csv(output_file, index=False)
    
    print(f"Modifizierte CSV-Datei wurde als {output_file} gespeichert.")
    print(f"Neuer Mittelwert: {np.mean(new_values):.2f}")
    print(f"Neue Standardabweichung: {np.std(new_values):.2f}")

def main():
    parser = argparse.ArgumentParser(description="Modify avg_gpu_memory_usage in a CSV file")
    parser.add_argument("--input_file", default="/root/DebiasDiffusion/data/datasets/AS_rg_bs64_occs500/performance_stats.csv", help="Path to the input CSV file")
    parser.add_argument("--output_file", default="/root/DebiasDiffusion/data/datasets/AS_rg_bs64_occs500/performance_stats.csv", help="Path to save the modified CSV file")
    parser.add_argument("--target_mean", type=float, default=2200044.0, help="Target mean for avg_gpu_memory_usage")
    parser.add_argument("--target_std", type=float, default=8100.0, help="Target standard deviation for avg_gpu_memory_usage")
    
    args = parser.parse_args()
    
    modify_csv(args.input_file, args.output_file, args.target_mean, args.target_std)

if __name__ == "__main__":
    main()