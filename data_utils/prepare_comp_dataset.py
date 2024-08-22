import os
import json

def process_dataset(input_dir, output_dir):
    # Define possible dataset files
    dataset_files = {
        "val": {
            "challenges_file": os.path.join(input_dir, 'arc-agi_evaluation_challenges.json'),
            "solutions_file": os.path.join(input_dir, 'arc-agi_evaluation_solutions.json'),
            "output_dir": os.path.join(output_dir, 'val')
        },
        "test": {
            "challenges_file": os.path.join(input_dir, 'arc-agi_test_challenges.json'),
            "solutions_file": None,  # No solutions file for test
            "output_dir": os.path.join(output_dir, 'test')
        },
        "train": {
            "challenges_file": os.path.join(input_dir, 'arc-agi_training_challenges.json'),
            "solutions_file": os.path.join(input_dir, 'arc-agi_training_solutions.json'),
            "output_dir": os.path.join(output_dir, 'train')
        }
    }

    # Process each dataset type
    for dataset_type, paths in dataset_files.items():
        challenges_file = paths["challenges_file"]
        solutions_file = paths["solutions_file"]
        dataset_output_dir = paths["output_dir"]

        # Check if the challenges file exists before processing
        if not os.path.exists(challenges_file):
            print(f"Skipping {dataset_type} dataset: challenges file not found.")
            continue

        # Load the challenges JSON data
        with open(challenges_file, 'r') as f:
            challenges_data = json.load(f)

        # Load the solutions JSON data, if it exists
        solutions_data = None
        if solutions_file and os.path.exists(solutions_file):
            with open(solutions_file, 'r') as f:
                solutions_data = json.load(f)

        # Ensure output directory exists
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Iterate over each task in the challenges data
        for task_id, task_data in challenges_data.items():
            # Create a directory for the current task
            task_dir = os.path.join(dataset_output_dir, task_id)
            os.makedirs(task_dir, exist_ok=True)

            # For val or train sets, process train and test data with solutions
            if dataset_type in ["val", "train"] and solutions_data:
                # Prepare train_input_output.json
                train_data = task_data.get('train', [])
                train_json_path = os.path.join(task_dir, 'train_input_output.json')
                with open(train_json_path, 'w') as train_json_file:
                    json.dump({'train': train_data}, train_json_file, indent=4)

                # Prepare test_input_output.json
                test_data = task_data.get('test', [])
                solution_data = solutions_data.get(task_id, [])
                test_json_path = os.path.join(task_dir, 'test_input_output.json')
                with open(test_json_path, 'w') as test_json_file:
                    json.dump({'test': test_data, 'solution': solution_data}, test_json_file, indent=4)

            # For test set, process only test data without solutions
            elif dataset_type == "test":
                # Prepare test_input.json
                test_data = task_data.get('test', [])
                test_json_path = os.path.join(task_dir, 'test_input.json')
                with open(test_json_path, 'w') as test_json_file:
                    json.dump({'test': test_data}, test_json_file, indent=4)

        print(f"{dataset_type.capitalize()} dataset processing complete. JSON files have been saved.")

if __name__ == "__main__":
    input_directory = "./arc-prize"  # You can change this to your actual input directory
    output_directory = "./arc_prize_processed"  # Directory where processed data will be saved

    process_dataset(input_directory, output_directory)