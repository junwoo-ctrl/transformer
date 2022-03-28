import sentencepiece as spm

vocab_file = "kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

lines = [
  "김준호는 사상 최악의 인재임에 틀림없다.",
  "폭주하는 김기동을 막을 수 있는 자가 누가 있으랴."
]
for line in lines:
  pieces = vocab.encode_as_pieces(line)
  ids = vocab.encode_as_ids(line)
  print(line)
  print(pieces)
  print(ids)
  print()