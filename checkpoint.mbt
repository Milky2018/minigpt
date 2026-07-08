///|
const CHECKPOINT_VERSION : Int = 1

///|
pub struct MiniGPTCheckpoint {
  version : Int
  vocabulary : Array[String]
  n_embd : Int
  token_embedding_table : Array[Double]
  lm_head : Array[Double]
} derive(ToJson, FromJson)

///|
pub fn MiniGPTCheckpoint::from_model(
  model : MiniGPT,
  tokenizer : CharTokenizer,
) -> MiniGPTCheckpoint {
  {
    version: CHECKPOINT_VERSION,
    vocabulary: tokenizer.vocabulary(),
    n_embd: model.n_embd(),
    token_embedding_table: model.token_embedding_data(),
    lm_head: model.lm_head_data(),
  }
}

///|
pub fn MiniGPTCheckpoint::to_model_and_tokenizer(
  self : MiniGPTCheckpoint,
) -> (MiniGPT, CharTokenizer) {
  if self.version != CHECKPOINT_VERSION {
    abort("unsupported MiniGPT checkpoint version")
  }
  let tokenizer = CharTokenizer::from_vocabulary(self.vocabulary)
  let model = MiniGPT::from_parameters(
    tokenizer.vocab_size(),
    self.n_embd,
    self.token_embedding_table,
    self.lm_head,
  )
  (model, tokenizer)
}

///|
pub fn encode_checkpoint(model : MiniGPT, tokenizer : CharTokenizer) -> String {
  MiniGPTCheckpoint::from_model(model, tokenizer).to_json().stringify(indent=2)
}

///|
pub fn decode_checkpoint(text : String) -> (MiniGPT, CharTokenizer) raise {
  let json = @json.parse(text[:])
  let checkpoint : MiniGPTCheckpoint = @json.from_json(json)
  checkpoint.to_model_and_tokenizer()
}
