Ashim Dhital

Screenplay Transformer
======================

A transformer language model I built from scratch that generates movie
screenplays. Takes a natural language prompt like "dark alley at night"
and outputs properly formatted screenplay text with scene headings,
character names, dialog, and action descriptions.

Built this to understand how these models actually work - tokenization,
attention, training dynamics, etc. - instead of just calling an API.


How to Build
------------
    pip install -r requirements.txt

Requirements: torch, regex, gradio (for web app)


Files
-----
- screenplay_transformer.py : the whole model - tokenizer, transformer, training, generation
- clean_scripts.py          : cleans raw screenplay data for training
- prepare_scenes.py         : splits scripts into individual scenes
- app.py                    : gradio web interface for deployment
- requirements.txt          : dependencies


How to Run (Full Pipeline)
--------------------------

1. Download the dataset from Kaggle:

    kaggle datasets download -d gufukuro/movie-scripts-corpus
    unzip movie-scripts-corpus.zip -d ./screenplay_data

2. Clean the raw scripts:

    python clean_scripts.py \
        --input_dir ./screenplay_data/screenplay_data/data \
        --output_dir ./cleaned_data

3. Prepare scene-level training data:

    python prepare_scenes.py \
        --input ./cleaned_data/training_data.txt \
        --output ./cleaned_data/scenes

4. Train the model:

    python screenplay_transformer.py train \
        --data ./cleaned_data/scenes/scene_training_data.txt \
        --tokenizer ./tokenizer \
        --output ./checkpoints \
        --epochs 10 \
        --batch 32

5. Generate a scene:

    python screenplay_transformer.py generate \
        --model ./checkpoints/best.pt \
        --tokenizer ./tokenizer \
        --prompt "dark alley at night" \
        --scene

6. Interactive mode:

    python screenplay_transformer.py interactive \
        --model ./checkpoints/best.pt \
        --tokenizer ./tokenizer


Model Architecture
------------------
Decoder-only transformer following the GPT pattern:

- Token embedding + learned positional embedding
- 8 transformer blocks, each with:
  - Multi-head self-attention (8 heads, causal masking)
  - Feed-forward network (4x expansion, GELU activation)
  - Pre-layer normalization, residual connections
- Final layer norm -> linear projection to vocab

~26M parameters with vocab size 10000 and embedding dim 512.


Tokenizer
---------
Byte-Pair Encoding (BPE) implemented from scratch. Starts with 256
byte-level tokens, iteratively merges the most frequent pairs until
hitting vocab size. Uses the GPT-2 regex pattern for word splitting.

Special tokens for screenplay structure:
- <SCENE>, </SCENE>       : scene headings (INT. OFFICE - DAY)
- <CHARACTER>, </CHARACTER> : character names
- <DIALOG>, </DIALOG>     : dialog lines
- <ACTION>, </ACTION>     : action/description
- <SCENE_START>, <SCENE_END> : scene boundaries for training


Training
--------
- Loss: cross-entropy (next token prediction)
- Optimizer: AdamW with weight decay
- LR schedule: linear warmup then cosine decay
- Gradient clipping to prevent exploding gradients

On Apple M4 (MPS): ~0.3-0.5 sec per step
On NVIDIA GPU (Colab): ~0.05-0.1 sec per step

With 108M tokens of training data, expect 6-10 hours on M4 or
1-2 hours on a GPU for good results.


Generation
----------
Autoregressive generation with:
- Temperature scaling (lower = more deterministic)
- Top-k filtering (only sample from k most likely tokens)
- Top-p / nucleus sampling (sample from smallest set with cumulative prob >= p)
- Optional stop token to generate complete scenes


Deployment
----------
The gradio app (app.py) can be deployed to Hugging Face Spaces:

1. Create a new Space at huggingface.co/spaces (Gradio SDK)
2. Upload these files:
   - app.py
   - screenplay_transformer.py
   - requirements.txt
   - tokenizer/tokenizer.json
   - checkpoints/best.pt
3. Done - you get a public URL

Test locally first:

    python app.py
    # Opens at http://localhost:7860


Example Output
--------------
Prompt: "dark alley at night"

    EXT. DARK ALLEY - NIGHT

    Rain drips from fire escapes. A figure emerges from
    the shadows.

                             DETECTIVE HARRIS
                   Someone was here. Recently.

    He kneels, examines a cigarette butt on the wet pavement.

                             DETECTIVE HARRIS
                   Still warm.


Notes
-----
- Training data: ~425MB of screenplay text (~108M tokens)
- The model learns screenplay formatting from the special tokens
- Scene-level training works better than full scripts (fits context window)
- Loss around 2-3 gives coherent output; 5+ gives gibberish
- Used PyTorch, no other ML frameworks
