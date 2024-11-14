# SoftmaxOutputInLlama3.1

## About
This is a quick script to get the output tenosrs of the softmax computation in llama3.1 and see how sparse the tensors can be

Basically, all this code does is
1. load the Llama model
2. replace the default softmax function with a custom one, where the custom softmax 
    - does the same softmax computation as the default one
    - gets the output tensor of the computation
    - gets the ratio of the non-zero values in the output tensor
    - writes it in a file called `outputSoftmaxNonZeroRatio.txt`
3. read in the input from `input.txt`
4. run the llama model 
5. writes the response from the model to `response.txt`
5. outputs the summary of the results

## Requirements
### Packages 
- transformers
- torch
### Models
You can either get the model from HuggingFace or Meta. As long as you have access to the model, you are good to go.

## How to run it
### Loading the inputs
Make sure you have the input you want to feed the model saved in the `input.txt` file.

There are some template questions in the `Inputs` directory that you can try out. But feel free to try other inputs to see how the results can be changed.

### Setting up for the models
there are two methods this script can get the model
1. loading the model from the `~/.cache/huggingface/hub` directory (assuming you have HuggingFace set up)
2. loading the model from your local directory (assuming you have downloaded the model and stored it somewhere)

If you choose the first method, there is no need to modify the code as it is all handled in the script

If you choose the second method, make sure to change the path in the code to match your local settings (search for the `TODO` tag in the script to update this)

> [!NOTE]
> do keep in mind that with the first method, the process of loading the model will be faster, as there are less overheads and that it is optimized

### Running the script
Once you finish all the steps above, simply run
```
python3 outputSoftmaxNonZeroRatio.py
```

Once the GPU is setup, the script will prompt you to choose the method you wish to load the model. For more details, read the prompt and the "Setting up for the models" section above

After choosing the method, the script will run and your outputs will be saved in `response.txt` and `outputSoftmaxNonZeroRatio.txt`, with a summary of the results in the terminal

