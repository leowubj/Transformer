
import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, name):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)


        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)
        input_tensor = input_tensor.to(device)


        # Process the input tensor through the encoder model
        _,  attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps



        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):

            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = np.sum(att_map, axis=1)

            if np.any(total_prob_over_rows < 0.99) or np.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows)

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"{name} Attention Map {j + 1}")
            
            # Save the plot
            plt.savefig(f"{name} attention_map_{j + 1}.png")
            
            # Show the plot
            plt.show()
            


