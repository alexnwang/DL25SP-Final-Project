import os
import json
import subprocess
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

def find_latest_checkpoint(run_dir):
    """Finds the checkpoint file with the highest epoch number."""
    checkpoints = glob.glob(os.path.join(run_dir, 'model_epoch_*.pth'))
    if not checkpoints:
        # Fallback if no epoch number pattern matches
        checkpoints = glob.glob(os.path.join(run_dir, '*.pth'))
        if not checkpoints:
             return None

    latest_checkpoint = None
    max_epoch = -1

    # Try to extract epoch number
    epoch_pattern = re.compile(r'model_epoch_(\d+)\.pth')
    found_with_epoch = False
    for ckpt in checkpoints:
        match = epoch_pattern.search(os.path.basename(ckpt))
        if match:
            found_with_epoch = True
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_checkpoint = ckpt

    if found_with_epoch:
        return latest_checkpoint
    elif checkpoints:
         # If no epoch numbers found, sort alphabetically and take the last one
         # This provides some deterministic behavior if multiple non-epoch checkpoints exist
        checkpoints.sort()
        latest_checkpoint = checkpoints[-1]
        print(f"Warning: Could not determine epoch number for checkpoints in {run_dir}. Using {os.path.basename(latest_checkpoint)}")
        return latest_checkpoint
    else:
        # Should not happen if checkpoints list was not empty initially, but included for safety
        return None


def parse_losses(output):
    """Parses the output of main.py to extract losses."""
    losses = {}
    # Regex to find lines like "key loss: value"
    loss_pattern = re.compile(r"^\s*([\w\s]+?)\s+loss:\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)\s*$", re.IGNORECASE)
    for line in output.splitlines():
        match = loss_pattern.match(line)
        if match:
             loss_name = match.group(1).strip()
             try:
                 loss_value = float(match.group(2))
                 losses[loss_name] = loss_value
             except ValueError:
                  print(f"Warning: Could not parse float from loss value in line: {line}")
                  continue # Skip if conversion fails
    return losses

def map_hyperparams_to_args(hyperparams, checkpoint_path):
    """Maps hyperparameters from JSON to command line arguments for main.py."""
    args_list = ["python", "main.py", "--checkpoint", checkpoint_path]

    # Mapping from hyperparams.json keys to main.py argument names
    # Add more mappings if needed based on hyperparams.json content
    mapping = {
        "encoder_name": "--encoder-name",
        "feature_key": "--feature-key",
        "num_hist": "--num-hist",
        "pred_depth": "--predictor-depth",
        "pred_heads": "--predictor-heads",
        "pred_dim_head": "--predictor-dim-head",
        "pred_mlp_dim": "--predictor-mlp-dim",
        "pred_dropout": "--predictor-dropout",
        "pred_emb_dropout": "--predictor-emb-dropout",
        "pred_pool": "--predictor-pool",
        # Add other potential mappings from hyperparams.json if they exist
        # e.g., "learning_rate": "--lr" (if main.py accepted --lr)
    }

    for key, arg_name in mapping.items():
        if key in hyperparams:
            # Skip adding if the value is None or empty string, maybe default is desired
            if hyperparams[key] is not None and hyperparams[key] != '':
                 args_list.extend([arg_name, str(hyperparams[key])])
        # else:
            # Optional: Warn if a hyperparameter expected by main.py's mapping is missing
            # print(f"Note: Hyperparameter '{key}' not found in hyperparams.json for {checkpoint_path}")

    return args_list


def main(runs_base_dir):
    results = {}

    # Find all hyperparams.json files recursively within the specified base directory
    hyperparam_files = glob.glob(os.path.join(runs_base_dir, '**', 'hyperparams.json'), recursive=True)

    if not hyperparam_files:
        print(f"Error: No hyperparams.json files found under {runs_base_dir}")
        return

    print(f"Found {len(hyperparam_files)} hyperparameter files. Processing runs...")

    for hp_file in hyperparam_files:
        run_dir = os.path.dirname(hp_file)
        # Try to get a more descriptive run name, perhaps parent of run_dir if nested deeper
        relative_run_path = os.path.relpath(run_dir, runs_base_dir)
        run_name = relative_run_path.replace(os.sep, '_') # Use relative path as ID

        print(f"\nProcessing run: {run_name} ({run_dir})")

        # Find the latest checkpoint
        checkpoint_path = find_latest_checkpoint(run_dir)
        if not checkpoint_path:
            print(f"  Skipping run {run_name}: No checkpoint file found.")
            results[run_name] = {"error": "No checkpoint found"}
            continue
        # Use absolute path for checkpoint for robustness
        checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"  Using checkpoint: {checkpoint_path}")

        # Load hyperparameters
        try:
            with open(hp_file, 'r') as f:
                hyperparams = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Skipping run {run_name}: Error reading {hp_file}: {e}")
            results[run_name] = {"error": f"JSON decode error in {os.path.basename(hp_file)}"}
            continue
        except FileNotFoundError:
             print(f"  Skipping run {run_name}: File not found {hp_file}.") # Should not happen with glob
             results[run_name] = {"error": f"File not found {os.path.basename(hp_file)}"}
             continue

        # Construct command
        command_args = map_hyperparams_to_args(hyperparams, checkpoint_path)
        print(f"  Running command: {' '.join(command_args)}")

        # Run main.py and capture output
        try:
            # Using capture_output=True requires Python 3.7+
            # text=True decodes stdout/stderr as text
            # Set cwd to the script's directory to ensure main.py is found if it's relative
            script_dir = os.path.dirname(os.path.realpath(__file__))
            process = subprocess.run(command_args, capture_output=True, text=True, check=True, cwd=script_dir, timeout=600) # Added timeout
            output = process.stdout
            print(f"  main.py for {run_name} finished successfully.")
            # print("----- Output -----")
            # print(output) # Uncomment for debugging main.py output
            # print("------------------")

            # Parse losses
            run_losses = parse_losses(output)
            if not run_losses:
                 print(f"  Warning: No losses parsed for run {run_name}.")
                 # Check stderr for potential issues even if process didn't error
                 if process.stderr:
                      print("  Stderr:")
                      print(process.stderr)
                 results[run_name] = {} # Store empty dict to indicate processing attempt without loss found
            else:
                 print(f"  Parsed losses: {run_losses}")
                 results[run_name] = run_losses

        except subprocess.CalledProcessError as e:
            print(f"  Error running main.py for {run_name}:")
            print(f"  Return code: {e.returncode}")
            print("  Stdout:")
            print(e.stdout) # Print stdout on error too
            print("  Stderr:")
            print(e.stderr)
            results[run_name] = {"error": f"main.py failed with code {e.returncode}"}
        except subprocess.TimeoutExpired as e:
             print(f"  Error running main.py for {run_name}: Timeout expired ({e.timeout}s)")
             print("  Stdout so far:")
             print(e.stdout)
             print("  Stderr so far:")
             print(e.stderr)
             results[run_name] = {"error": "Timeout expired"}
        except FileNotFoundError:
             print(f"  Error: 'python' command not found. Make sure Python is installed and in PATH.")
             # Stop processing if python isn't found
             return
        except Exception as e:
             print(f"  An unexpected error occurred during processing of {run_name}: {e}")
             results[run_name] = {"error": f"Unexpected error: {str(e)}"}


    # --- Plotting Results ---
    print("\nPlotting results...")

    # Filter out runs that had errors or produced no loss values
    valid_results = {name: losses for name, losses in results.items()
                     if losses and "error" not in losses and losses} # Ensure losses dict is not empty

    if not valid_results:
        print("No valid results with losses found to plot.")
        print("\nSummary of encountered errors:")
        error_count = 0
        for name, data in results.items():
            if "error" in data:
                print(f"- {name}: {data['error']}")
                error_count += 1
            elif not data: # Empty dict means processed but no losses found
                 print(f"- {name}: Processed but no losses found in output.")
                 error_count +=1
        if error_count == 0 and results:
             print("All runs processed, but no loss values could be extracted from the output of main.py.")
        elif not results:
             print("No runs were processed at all.") # Should not happen if hyperparam_files was populated
        return

    run_names = list(valid_results.keys())
    # Dynamically get all unique loss keys found across all successful runs
    loss_keys = sorted(list(set(key for losses in valid_results.values() for key in losses)))

    if not loss_keys:
        print("Valid runs were processed, but no loss keys were found in the results.")
        return

    print(f"Plotting loss keys: {loss_keys}")

    num_runs = len(run_names)
    num_loss_types = len(loss_keys)
    # Adjust bar width and figure size dynamically
    bar_width = max(0.1, 0.8 / num_loss_types) # Ensure minimum width
    group_width = bar_width * num_loss_types
    # Calculate necessary figure width: provide space for labels + space per run group
    fig_width = max(10, num_runs * (group_width + 0.2) + 4) # Base width + width per run + margin
    fig_height = 8 # Increased height for potentially rotated labels

    index = np.arange(num_runs) * (group_width + 0.4) # Add more spacing between groups

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for i, loss_key in enumerate(loss_keys):
        # Get loss values, using np.nan if a specific loss key is missing for a run
        loss_values = [valid_results[name].get(loss_key, np.nan) for name in run_names]
        # Calculate bar positions relative to the center of the group
        bar_positions = index - (group_width / 2) + (i * bar_width) + (bar_width / 2)
        bars = ax.bar(bar_positions, loss_values, bar_width, label=loss_key)
        # Optional: Add text labels on bars if needed (can get crowded)
        # ax.bar_label(bars, padding=3, fmt='%.3f', rotation=90, size=8)


    ax.set_xlabel('Run Identifier')
    ax.set_ylabel('Loss')
    ax.set_title('Comparison of Probing Losses Across Runs')
    ax.set_xticks(index) # Set tick positions to the center of each group
    ax.set_xticklabels(run_names, rotation=60, ha="right", fontsize=9) # Rotate labels more if needed
    ax.legend(title="Loss Types")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Adjust y-axis limit if values are very small or large
    all_loss_vals = [val for losses in valid_results.values() for val in losses.values() if isinstance(val, (int, float))]
    if all_loss_vals:
         min_loss = min(all_loss_vals)
         max_loss = max(all_loss_vals)
         ax.set_ylim(bottom=min(0, min_loss * 1.1), top=max_loss * 1.1) # Add some padding

    plt.tight_layout() # Adjust layout to prevent labels overlapping

    plot_filename = "probing_losses_comparison.png"
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved to {os.path.abspath(plot_filename)}")
        # plt.show() # Uncomment to display the plot interactively
    except Exception as e:
        print(f"Error saving plot: {e}")

    print("\nSummary of runs with errors or missing losses:")
    error_count = 0
    for name, data in results.items():
         if "error" in data:
             print(f"- {name}: {data['error']}")
             error_count += 1
         elif not data:
              print(f"- {name}: Processed but no losses found.")
              error_count += 1
    if error_count == 0:
         print("All runs processed successfully and losses were extracted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JEPA models from multiple runs.")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/",
        help="Base directory containing the run folders (e.g., 'runs/').",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.runs_dir):
        print(f"Error: Specified runs directory '{args.runs_dir}' does not exist.")
    else:
        main(args.runs_dir) 