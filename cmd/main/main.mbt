///|
const DEFAULT_DATA_PATH : String = "data/tiny_shakespeare.txt"

///|
const DEFAULT_TOKENIZER_KIND : String = "char"

///|
const DEFAULT_PROMPT : String = "ROMEO:"

///|
const DEFAULT_MAX_NEW_TOKENS : Int = 80

///|
const DEFAULT_TOP_K : Int = 20

///|
const DEFAULT_TEMPERATURE : Double = 0.8

///|
const DEFAULT_MODEL_PATH : String = "minigpt-model.bin"

///|
fn cli_command() -> @argparse.Command {
  Command("minigpt", about="MiniGPT teaching demo", subcommand_required=true, subcommands=[
    Command(
      "train",
      about="train a fresh model on the Tiny Shakespeare corpus",
      options=training_options(),
    ),
    Command(
      "generate",
      about="sample text from a trained model checkpoint",
      options=generate_options(),
    ),
  ])
}

///|
fn training_options() -> Array[@argparse.OptionArg] {
  [
    OptionArg("data", long="data", about="UTF-8 training corpus path", default_values=[
      DEFAULT_DATA_PATH,
    ]),
    OptionArg(
      "tokenizer",
      long="tokenizer",
      about="tokenizer kind: char or word",
      default_values=[DEFAULT_TOKENIZER_KIND],
    ),
    OptionArg("out", long="out", about="model checkpoint output path", default_values=[
      DEFAULT_MODEL_PATH,
    ]),
    OptionArg("steps", long="steps", about="training steps", default_values=[
      @minigpt.TRAINING_STEPS.to_string(),
    ]),
    OptionArg("batch-size", long="batch-size", about="batch size", default_values=[
      @minigpt.BATCH_SIZE.to_string(),
    ]),
    OptionArg("block-size", long="block-size", about="context length", default_values=[
      @minigpt.BLOCK_SIZE.to_string(),
    ]),
    OptionArg("n-embd", long="n-embd", about="transformer embedding width", default_values=[
      @minigpt.N_EMBD.to_string(),
    ]),
    OptionArg("n-head", long="n-head", about="attention head count", default_values=[
      @minigpt.N_HEAD.to_string(),
    ]),
    OptionArg("n-layer", long="n-layer", about="transformer block count", default_values=[
      @minigpt.N_LAYER.to_string(),
    ]),
    OptionArg(
      "learning-rate",
      long="learning-rate",
      about="optimizer learning rate",
      default_values=[@minigpt.LEARNING_RATE.to_string()],
    ),
    OptionArg(
      "eval-interval",
      long="eval-interval",
      about="evaluate every N iterations",
      default_values=[@minigpt.EVAL_INTERVAL.to_string()],
    ),
    OptionArg(
      "eval-iters",
      long="eval-iters",
      about="number of batches per evaluation",
      default_values=[@minigpt.EVAL_ITERS.to_string()],
    ),
    OptionArg(
      "log-interval",
      long="log-interval",
      about="print train loss every N iterations",
      default_values=[@minigpt.LOG_INTERVAL.to_string()],
    ),
    OptionArg(
      "always-save-checkpoint",
      long="always-save-checkpoint",
      about="save at every eval after iter 0",
      default_values=["false"],
    ),
  ]
}

///|
fn generate_options() -> Array[@argparse.OptionArg] {
  [
    OptionArg("model", long="model", about="model checkpoint path", default_values=[
      DEFAULT_MODEL_PATH,
    ]),
    OptionArg("prompt", long="prompt", about="prompt text", default_values=[
      DEFAULT_PROMPT,
    ]),
    OptionArg(
      "max-new-tokens",
      long="max-new-tokens",
      about="number of generated characters",
      default_values=[DEFAULT_MAX_NEW_TOKENS.to_string()],
    ),
    OptionArg("top-k", long="top-k", about="sampler top-k", default_values=[
      DEFAULT_TOP_K.to_string(),
    ]),
    OptionArg("temperature", long="temperature", about="sampler temperature", default_values=[
      DEFAULT_TEMPERATURE.to_string(),
    ]),
  ]
}

///|
fn value(matches : @argparse.Matches, name : String) -> String {
  match matches.values.get(name) {
    Some(values) if values.length() > 0 => values[0]
    _ => abort("missing parsed argument: \{name}")
  }
}

///|
fn int_value(matches : @argparse.Matches, name : String) -> Int {
  @string.parse_int(value(matches, name)[:]) catch {
    _ => abort("invalid integer for --\{name}")
  }
}

///|
fn double_value(matches : @argparse.Matches, name : String) -> Double {
  @string.parse_double(value(matches, name)[:]) catch {
    _ => abort("invalid number for --\{name}")
  }
}

///|
fn bool_value(matches : @argparse.Matches, name : String) -> Bool {
  match value(matches, name) {
    "true" | "1" | "yes" => true
    "false" | "0" | "no" => false
    _ => abort("invalid boolean for --\{name}")
  }
}

///|
fn training_config(matches : @argparse.Matches) -> @minigpt.TrainingConfig {
  @minigpt.TrainingConfig::new(
    int_value(matches, "batch-size"),
    int_value(matches, "block-size"),
    int_value(matches, "steps"),
    double_value(matches, "learning-rate"),
    eval_interval=int_value(matches, "eval-interval"),
    eval_iters=int_value(matches, "eval-iters"),
    log_interval=int_value(matches, "log-interval"),
    always_save_checkpoint=bool_value(matches, "always-save-checkpoint"),
  )
}

///|
fn parse_tokenizer_kind(matches : @argparse.Matches) -> String {
  match value(matches, "tokenizer") {
    "char" => "char"
    "word" => "word"
    other => abort("unsupported tokenizer kind: \{other}")
  }
}

///|
fn load_dataset(
  path : String,
  tokenizer_kind : String,
) -> @tokenizer.TokenDataset {
  let text = @fs.read_file_to_string(path, encoding="utf8") catch {
    _ => abort("failed to read corpus '\{path}'")
  }
  @tokenizer.prepare_token_dataset(text, tokenizer_kind)
}

///|
fn print_loss_summary(stats : @minigpt.TrainingStats) -> Unit {
  println("initial eval loss: \{stats.initial_eval_loss}")
  println("final eval loss: \{stats.final_eval_loss}")
  println("best val loss: \{stats.best_val_loss}")
  println("saved checkpoints: \{stats.saved_checkpoints}")
  if stats.losses.length() == 0 {
    println("losses: []")
  } else {
    println("first train loss: \{stats.losses[0]}")
    println("last train loss: \{stats.losses[stats.losses.length() - 1]}")
  }
}

///|
fn print_completion_trace(
  model : @minigpt.MiniGPT,
  tokenizer : @tokenizer.Tokenizer,
  prompt : String,
  max_new_tokens : Int,
  top_k : Int,
  temperature : Double,
  rng : @random.Rand,
) -> Unit {
  let token_ids = tokenizer.encode(prompt) catch {
    UnknownCharacter(ch) =>
      abort("prompt contains character outside tokenizer vocabulary: \{ch}")
    UnknownToken(token) =>
      abort("prompt contains token outside tokenizer vocabulary: \{token}")
    InvalidTokenId(_) =>
      abort("unexpected token id error while encoding prompt")
  }
  println("completion:")
  println(prompt)
  for _ in 0..<max_new_tokens {
    let next_id = @minigpt.generate_next_token_id_from_context_with_sampling_options(
      model,
      token_ids,
      rng,
      top_k,
      temperature,
      None,
      None,
    )
    token_ids.push(next_id)
    let text = tokenizer.decode(token_ids) catch {
      _ => abort("failed to decode generated text")
    }
    println(text)
  }
}

///|
fn run_train(matches : @argparse.Matches) -> Unit {
  let data_path = value(matches, "data")
  let tokenizer_kind = parse_tokenizer_kind(matches)
  let out_path = value(matches, "out")
  let config = training_config(matches)
  let n_embd = int_value(matches, "n-embd")
  let n_head = int_value(matches, "n-head")
  let n_layer = int_value(matches, "n-layer")
  let rng = @random.Rand::new()
  let dataset = load_dataset(data_path, tokenizer_kind)
  println("command: train")
  println("data: \{data_path}")
  println("tokenizer: \{dataset.tokenizer.kind_name()}")
  println("vocab size: \{dataset.tokenizer.vocab_size()}")
  println("train tokens: \{dataset.train_ids.length()}")
  println("val tokens: \{dataset.val_ids.length()}")
  println("steps: \{config.steps}")
  println("batch size: \{config.batch_size}")
  println("block size: \{config.block_size}")
  println("learning rate: \{config.learning_rate}")
  println("eval interval: \{config.eval_interval}")
  println("eval iters: \{config.eval_iters}")
  println("log interval: \{config.log_interval}")
  println("always save checkpoint: \{config.always_save_checkpoint}")
  println("checkpoint: \{out_path}")
  let model = @minigpt.MiniGPT::new_gpt_with_options(
    dataset.tokenizer.vocab_size(),
    n_embd,
    n_head,
    n_layer,
    config.block_size,
    rng,
  )
  println("model kind: \{model.kind_name()}")
  println("n_embd: \{model.n_embd()}")
  println("n_head: \{model.n_head()}")
  println("n_layer: \{model.n_layer()}")
  let saved_checkpoints : Ref[Int] = { val: 0 }
  let stats = @minigpt.train_token_ids_with_progress_callback(
    model,
    dataset.train_ids,
    dataset.val_ids,
    config,
    rng,
    fn(iter_num, train_loss, val_loss) {
      println(
        "step \{iter_num}: train loss \{train_loss}, val loss \{val_loss}",
      )
    },
    fn(iter_num, loss, learning_rate) {
      println("iter \{iter_num}: loss \{loss}, lr \{learning_rate}")
    },
    fn(model, optimizer, iter_num, best_val_loss, config) {
      let checkpoint = @minigpt.encode_training_checkpoint(
        model,
        dataset.tokenizer,
        optimizer,
        iter_num,
        best_val_loss,
        config,
      )
      @fs.write_bytes_to_file(out_path, checkpoint) catch {
        _ => abort("failed to write checkpoint '\{out_path}'")
      }
      saved_checkpoints.val += 1
      println("saving checkpoint to \{out_path}")
    },
  )
  print_loss_summary(stats)
  if saved_checkpoints.val == 0 {
    println("no checkpoint saved; val loss did not improve after iter 0")
  }
}

///|
fn run_generate(matches : @argparse.Matches) -> Unit {
  let model_path = value(matches, "model")
  let prompt = value(matches, "prompt")
  let max_new_tokens = int_value(matches, "max-new-tokens")
  let top_k = int_value(matches, "top-k")
  let temperature = double_value(matches, "temperature")
  let rng = @random.Rand::new()
  if !@fs.path_exists(model_path) {
    abort(
      "checkpoint '\{model_path}' not found; run `moon run --release cmd/main -- train --out \{model_path}` first",
    )
  }
  let checkpoint = @fs.read_file_to_bytes(model_path) catch {
    _ => abort("failed to read checkpoint '\{model_path}'")
  }
  let (model, tokenizer) = @minigpt.decode_checkpoint(checkpoint)
  println("command: generate")
  println("model: \{model_path}")
  println("vocab size: \{tokenizer.vocab_size()}")
  println("n_embd: \{model.n_embd()}")
  println("n_head: \{model.n_head()}")
  println("n_layer: \{model.n_layer()}")
  println("model kind: \{model.kind_name()}")
  println("top_k: \{top_k}")
  println("temperature: \{temperature}")
  print_completion_trace(
    model, tokenizer, prompt, max_new_tokens, top_k, temperature, rng,
  )
}

///|
fn run(matches : @argparse.Matches) -> Unit {
  match matches.subcommand {
    Some(("train", sub)) => run_train(sub)
    Some(("generate", sub)) => run_generate(sub)
    _ => abort("expected subcommand: train or generate")
  }
}

///|
fn main {
  let command = cli_command()
  let matches = command.parse() catch {
    err => {
      println("\{err}")
      println(command.render_help())
      return
    }
  }
  run(matches)
}
