from gradientai import Gradient
import os
os.environ['GRADIENT_ACCESS_TOKEN'] = "UwrAcG6a56iKHmLZFEN8tpVUwUn8ilx7"
os.environ['GRADIENT_WORKSPACE_ID'] = "c34032a4-3dfb-4ea5-be66-8ff505b08961_workspace"


def main():
    with Gradient() as gradient:
        base_model = gradient.get_base_model(base_model_slug="llama2-7b-chat")
        new_model_adapter = base_model.create_model_adapter(
            name="test model 3")
        print(f"Created model adapter with id {new_model_adapter.id}")
        sample_query = "### Instruction: What does akhil teach? \n\n### Response: akhil teaches deep learning"
        print(f"Asking: {sample_query}")

        completion = new_model_adapter.complete(
            query=sample_query, max_generated_token_count=100).generated_output
        print(f"Generated(before fine tune): {completion}")
        samples = [
            {
                "inputs": "### Instruction: What does akhil teach? \n\n### Response: akhil teaches deep learning",

            },
            {
                "inputs": "### Instruction: What is taught ob akhil channel? \n\n### Response: akhil teaches deep learning",

            }
        ]
        num_epochs = 3
        count = 0
        while count < num_epochs:
            print(f"Fine tuning the model {count+1}/{num_epochs}")
            new_model_adapter.fine_tune(samples=samples)
            count += 1

        completion = new_model_adapter.complete(
            query=sample_query, max_generated_token_count=100).generated_output
        print(f"Generated(after fine tune): {completion}")

        new_model_adapter.delete()

        if __name__ == "__main__":
            main()
