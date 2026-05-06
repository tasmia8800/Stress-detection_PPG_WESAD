import subprocess
import sys
import time
import os
import argparse

# Set working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def run_step(script_name, description, env=None):
    print(f"==================================================")
    print(f"[{time.strftime('%H:%M:%S')}] Starting: {description}")
    print(f"Script: {script_name}")
    print(f"==================================================")
    
    start_time = time.time()
    
    cmd = [sys.executable, script_name]
    
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    try:
        subprocess.run(cmd, check=True, env=run_env)
        
        duration = time.time() - start_time
        print(f"\n[{time.strftime('%H:%M:%S')}] Finished: {description}")
        print(f"Duration: {duration:.2f} seconds")
        print("Status: SUCCESS\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] FAILED: {description}")
        print(f"Error executing {script_name}. Exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)


def main():
    print("==================================================")
    print(f"[{time.strftime('%H:%M:%S')}] Starting WESAD Stress Classification Pipeline")
    print("==================================================")
    
    pipeline_env = {'WESAD_MODE': 'ORIGINAL'}
    
    target_file = 'result/baseline_results.csv'
    if os.path.exists(target_file):
        print(f"[CLEANUP] Removing old results file: {target_file}")
        os.remove(target_file)

    start_time = time.time()

    run_step("read_data_new_binary.py", "Data Processing - Binary (Stress vs Non-Stress)", env=pipeline_env)
    run_step("read_data_new_tri.py", "Data Processing - 3-Class (Baseline, Stress, Amusement)", env=pipeline_env)
    run_step("read_data_new_quad.py", "Data Processing - 4-Class (Adding Meditation)", env=pipeline_env)

    run_step("ML_binary.py", "Classification - Binary", env=pipeline_env)
    run_step("ML_tri.py", "Classification - 3-Class", env=pipeline_env)
    run_step("ML_quad.py", "Classification - 4-Class", env=pipeline_env)

    run_step("plot_results.py", "Visualization - Generating Performance Plots", env=pipeline_env)

    total_duration = time.time() - start_time
    print(f"==================================================")
    print(f"Pipeline Completed Successfully!")
    print(f"Total Duration: {total_duration/60:.2f} minutes")
    print(f"==================================================")

if __name__ == "__main__":
    if os.path.exists("read_data_new_binary.py"):
        main()
    else:
        print("Error: Could not find project scripts. Please run this from the project root directory (d:\\Hissain\\github\\WESAD_Stress_Classification\\PPG-Based).")
