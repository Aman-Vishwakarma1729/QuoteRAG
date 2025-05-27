---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:25000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: to me fearless is not the absense of fear its not being completely
    unafraid to me fearless is having fears fearless is having doubts lots of them
    to me fearless is living in spite of those things that scare you to death
  sentences:
  - yesterday is gone tomorrow has not yet come we have only today let us begin
  - all our dreams can come true if we have the courage to pursue them
  - i can never read all the books i want i can never be all the people i want and
    live all the lives i want i can never train myself in all the skills i want and
    why do i want i want to live and feel all the shades tones and variations of mental
    and physical experience possible in my life and i am horribly limited
- source_sentence: happiness in intelligent people is the rarest thing i know
  sentences:
  - it is a great thing to start life with a small number of really good books which
    are your very own
  - friendship my definition is built on two things respect and trust both elements
    have to be there and it has to be mutual you can have respect for someone but
    if you dont have trust the friendship will crumble
  - and all the books youve read have been read by other people and all the songs
    youve loved have been heard by other people and that girl thats pretty to you
    is pretty to other people and that if you looked at these facts when you were
    happy you would feel great because you are describing unity
- source_sentence: in politics if you want anything said ask a man if you want anything
    done ask a woman
  sentences:
  - i know not with what weapons world war iii will be fought but world war iv will
    be fought with sticks and stones
  - i can believe things that are true and things that arent true and i can believe
    things where nobody knows if theyre true or not i can believe in santa claus and
    the easter bunny and the beatles and marilyn monroe and elvis and mister ed listen  i
    believe that people are perfectable that knowledge is infinite that the world
    is run by secret banking cartels and is visited by aliens on a regular basis nice
    ones that look like wrinkled lemurs and bad ones who mutilate cattle and want
    our water and our women i believe that the future sucks and i believe that the
    future rocks and i believe that one day white buffalo woman is going to come back
    and kick everyones ass i believe that all men are just overgrown boys with deep
    problems communicating and that the decline in good sex in america is coincident
    with the decline in drivein movie theaters from state to state i believe that
    all politicians are unprincipled crooks and i still believe that they are better
    than the alternative i believe that california is going to sink into the sea when
    the big one comes while florida is going to dissolve into madness and alligators
    and toxic waste i believe that antibacterial soap is destroying our resistance
    to dirt and disease so that one day well all be wiped out by the common cold like
    martians in war of the worlds i believe that the greatest poets of the last century
    were edith sitwell and don marquis that jade is dried dragon sperm and that thousands
    of years ago in a former life i was a onearmed siberian shaman i believe that
    mankinds destiny lies in the stars i believe that candy really did taste better
    when i was a kid that its aerodynamically impossible for a bumble bee to fly that
    light is a wave and a particle that theres a cat in a box somewhere whos alive
    and dead at the same time although if they dont ever open the box to feed it itll
    eventually just be two different kinds of dead and that there are stars in the
    universe billions of years older than the universe itself i believe in a personal
    god who cares about me and worries and oversees everything i do i believe in an
    impersonal god who set the universe in motion and went off to hang with her girlfriends
    and doesnt even know that im alive i believe in an empty and godless universe
    of causal chaos background noise and sheer blind luck i believe that anyone who
    says sex is overrated just hasnt done it properly i believe that anyone who claims
    to know whats going on will lie about the little things too i believe in absolute
    honesty and sensible social lies i believe in a womans right to choose a babys
    right to live that while all human life is sacred theres nothing wrong with the
    death penalty if you can trust the legal system implicitly and that no one but
    a moron would ever trust the legal system i believe that life is a game that life
    is a cruel joke and that life is what happens when youre alive and that you might
    as well lie back and enjoy it
  - everyone is a moon and has a dark side which he never shows to anybody
- source_sentence: failure is the condiment that gives success its flavor
  sentences:
  - i have not failed ive just found 10000 ways that wont work
  - if you remember me then i dont care if everyone else forgets
  - the worst type of crying wasnt the kind everyone could seethe wailing on street
    corners the tearing at clothes no the worst kind happened when your soul wept
    and no matter what you did there was no way to comfort it a section withered and
    became a scar on the part of your soul that survived for people like me and echo
    our souls contained more scar tissue than life
- source_sentence: but until a person can say deeply and honestly i am what i am today
    because of the choices i made yesterday that person cannot say i choose otherwise
  sentences:
  - sebastian just smiled i could hear your heart beating he said softly when you
    were watching me with valentine did it bother youthat you seem to be dating my
    dad jace shrugged youre a little young for him to be honestwhat for the first
    time since jace had met him sebastian seemed flabbergasted
  - i fell in love with him but i dont just stay with him by default as if theres
    no one else available to me i stay with him because i choose to every day that
    i wake up every day that we fight or lie to each other or disappoint each other
    i choose him over and over again and he chooses me
  - marriage can wait education cannot
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'but until a person can say deeply and honestly i am what i am today because of the choices i made yesterday that person cannot say i choose otherwise',
    'i fell in love with him but i dont just stay with him by default as if theres no one else available to me i stay with him because i choose to every day that i wake up every day that we fight or lie to each other or disappoint each other i choose him over and over again and he chooses me',
    'marriage can wait education cannot',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 25,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                          |
  | details | <ul><li>min: 5 tokens</li><li>mean: 31.97 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 33.06 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.28</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                        | sentence_1                                                                                                                                                                                                                                                                    | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>walk as if you are kissing the earth with your feet</code>                                                                                                                                                  | <code>it is a curious thing harry but perhaps those who are best suited to power are those who have never sought it those who like you have leadership thrust upon them and take up the mantle because they must and find to their own surprise that they wear it well</code> | <code>0.0</code> |
  | <code>you may encounter many defeats but you must not be defeated in fact it may be necessary to encounter the defeats so you can know who you are what you can rise from how you can still come out of it</code> | <code>failure is the condiment that gives success its flavor</code>                                                                                                                                                                                                           | <code>1.0</code> |
  | <code>there is only one thing that makes a dream impossible to achieve the fear of failure</code>                                                                                                                 | <code>fear doesnt shut you down it wakes you up</code>                                                                                                                                                                                                                        | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 0.16  | 500  | 0.163         |
| 0.32  | 1000 | 0.1426        |
| 0.48  | 1500 | 0.1367        |
| 0.64  | 2000 | 0.1309        |
| 0.8   | 2500 | 0.1212        |
| 0.96  | 3000 | 0.1302        |


### Framework Versions
- Python: 3.10.0
- Sentence Transformers: 4.1.0
- Transformers: 4.52.3
- PyTorch: 2.7.0+cu126
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->