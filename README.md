# Bi-directional Attention Flow for Machine Comprehension
 
- This the original implementation of [Bi-directional Attention Flow for Machine Comprehension][paper] (Seo et al., 2016).
- The CodaLab worksheet for the [SQuAD Leaderboard][squad] submission is available [here][worksheet].
- Please contact [Minjoon Seo][minjoon] ([@seominjoon][minjoon-github]) for questions and suggestions.

## 0. Requirements
#### General
- Python (verified on 3.5.2. Issues have been reported with Python 2!)
- unzip, wget (for running `download.sh` only)

#### Python Packages
- tensorflow (deep learning library, verified on r0.11)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visualization; if you only train and test, not needed)

## 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

## 2. Training
The model has 2,571,787 parameters.
The model was trained with NVidia Titan X (Pascal Architecture, 2016).
The model requires at least 12GB of GPU RAM.
If your GPU RAM is smaller than 12GB, you can either decrease batch size (performance might degrade),
or you can use multi GPU (see below).
The training converges at ~18k steps, and it took ~4s per step (i.e. ~20 hours).

Before training, it is recommended to first try the following code to verify everything is okay and memory is sufficient:
```
python -m basic.cli --mode train --noload --debug
```

Then to fully train, run:
```
python -m basic.cli --mode train --noload
```

You can speed up the training process with optimization flags:
```
python -m basic.cli --mode train --noload --len_opt --cluster
```
You can still omit them, but training will be much slower.


## 3. Test
To test, run:
```
python -m basic.cli
```

Similarly to training, you can give the optimization flags to speed up test (5 minutes on dev data):
```
python -m basic.cli --len_opt --cluster
```

This command loads the most recently saved model during training and begins testing on the test data.
After the process ends, it prints F1 and EM scores, and also outputs a json file (`$PWD/out/basic/00/answer/test-####.json`,
where `####` is the step # that the model was saved).
Note that the printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator (copied in `squad` folder) and the output json file:

```
python squad/evaluate-v1.1.py $HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-####.json
```

### 3.1 Loading from pre-trained weights
Instead of training the model yourself, you can choose to use pre-trained weights that were used for [SQuAD Leaderboard][squad] submission.
Refer to [this worksheet][worksheet] in CodaLab to reproduce the results.
If you are unfamiliar with CodaLab, follow these simple steps (given that you met all prereqs above):

1. Download `save.zip` from the [worksheet][worksheet] and unzip it in the current directory.
2. Copy `glove.6B.100d.txt` from your glove data folder (`$HOME/data/glove/`) to the current directory.
3. To reproduce single model:
  
  ```
  basic/run_single.sh $HOME/data/squad/dev-v1.1.json single.json
  ```
  
  This writes the answers to `single.json` in the current directory. You can then use the official evaluator to obtain EM and F1 scores. If you want to run on GPU (~5 mins), change the value of batch_size flag in the shell file to a higher number (60 for 12GB GPU RAM). 
4. Similarly, to reproduce ensemble method:
  
  ```
  basic/run_ensemble.sh $HOME/data/squad/dev-v1.1.json ensemble.json 
  ```
  If you want to run on GPU, you should run the script sequentially by removing '&' in the forloop, or you will need to specify different GPUs for each run of the for loop.

## Results

###Dev Data

|          | EM (%) | F1 (%) |
| -------- |:------:|:------:|
| single   | 67.7   | 77.3   |
| ensemble | 72.6   | 80.7   |

###Test Data

|          | EM (%) | F1 (%) |
| -------- |:------:|:------:|
| single   | 68.0   | 77.3   |
| ensemble | 73.3   | 81.1   |

Refer to [our paper][paper] for more details.
See [SQuAD Leaderboard][squad] to compare with other models.


<!--
## Using Pre-trained Model

If you would like to use pre-trained model, it's very easy! 
You can download the model weights [here][save] (make sure that its commit id matches the source code's).
Extract them and put them in `$PWD/out/basic/00/save` directory, with names unchanged.
Then do the testing again, but you need to specify the step # that you are loading from:
```
python -m basic.cli --mode test --batch_size 8 --eval_num_batches 0 --load_step ####
```
-->


## Multi-GPU Training & Testing
Our model supports multi-GPU training.
We follow the parallelization paradigm described in [TensorFlow Tutorial][multi-gpu].
In short, if you want to use batch size of 60 (default) but if you have 3 GPUs with 4GB of RAM,
then you initialize each GPU with batch size of 20, and combine the gradients on CPU.
This can be easily done by running:
```
python -m basic.cli --mode train --noload --num_gpus 3 --batch_size 20
```

Similarly, you can speed up your testing by:
```
python -m basic.cli --num_gpus 3 --batch_size 20 
```

## 4. Demo Server
Run demo server to explore a single trained model. Firstly, we need to preprocess the data.
```
python -m squad.prepro --mode dev --source_dir /path-to/dev-v1.1.json --target_dir inter_single --glove_dir /path-to/glove/
```
This will create a folder called `inter_single` in the current path. Before executing the next command, make sure your `PYTHONPATH` has the current project folder on it. You can set the path by `export PYTHONPATH='/path-to/bi-att-flow'`.
Then execute the following command to run the server: 
```
python -m demo.run --data_dir=/path-to/inter_single/ --load_path=/path-to/save/37/save --shared_path=/path-to/save/37/shared.json
```

Wait for everything to load and then type ` http://0.0.0.0:1995/` in your browser.
 
## 5. Inference Module

Run the preprocessing step from 4 if you haven't done it yet. After that, you can use the basic/inference module to predict answers from new text:
```
    python -m basic.inference --data_dir = "/path-to/inter_single/" --load_path = "/path-to/save/40/save" --shared_path = "/path-to/save/40/shared.json"
```
You can also import it from another module: 

````
    from basic.inference import Inference
    inference = Inference()
    context = 'More than 26,000 square kilometres (10,000 sq mi) of Victorian farmland are sown for grain, mostly in the state west. More than 50% of this area is sown for wheat, 33% is sown for barley and 7% is sown for oats. A further 6,000 square kilometres (2,300 sq mi) is sown for hay. In 2003–04, Victorian farmers produced more than 3 million tonnes of wheat and 2 million tonnes of barley. Victorian farms produce nearly 90% of Australian pears and third of apples. It is also a leader in stone fruit production. The main vegetable crops include asparagus, broccoli, carrots, potatoes and tomatoes. Last year, 121,200 tonnes of pears and 270,000 tonnes of tomatoes were produced.'
    question = 'What percentage of farmland grows wheat? '
    inference.predict(context, question)
````
This returns a tuple (predicted_anser, confidence). The confidence is the softmaxed logits
````
('50%', 0.582761824131012)
````

You can get rid of all the flags in 4 and 5 by hardcoding them in `inference.py`.


[multi-gpu]: https://www.tensorflow.org/versions/r0.11/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards
[squad]: http://stanford-qa.com
[paper]: https://arxiv.org/abs/1611.01603
[worksheet]: https://worksheets.codalab.org/worksheets/0x37a9b8c44f6845c28866267ef941c89d/
[minjoon]: https://seominjoon.github.io
[minjoon-github]: https://github.com/seominjoon
