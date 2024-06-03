---
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: true
      share: true
      related: true
    author: Ceaser
---

요사이 공부하였던 transformer에 대해 정리하고자 한다.   
그러나 그 전에 쓰였던 seq2seq, attnetion부터 정리하여 보자   

# 1. Sequence to Sequence   
- squence to sequence 논문:  <https://arxiv.org/abs/1409.3215>   
- 참조한 블로그: <https://wikidocs.net/24996>

sequence 2 sequence(seq2seq)는 Autoencoder와 비슷하다고 생각이 들었다.   
다음의 이미지를 보면 이해가 쉬울 것이다.   
![seq2seq_image](https://wikidocs.net/images/page/24996/%EC%9D%B8%EC%BD%94%EB%8D%94%EB%94%94%EC%BD%94%EB%8D%94%EB%AA%A8%EB%8D%B8.PNG)   
바로 위의 이미지는 sequence to sequence 이미지.   

<img src="https://velog.velcdn.com/images/jochedda/post/f01b86c5-5025-434b-b365-2e798a4f6538/image.png" width='60%' hight='70%' alt="Autoencoder_image">   
    
바로 위의 이미지는 Autoencoder 이미지.   

여기서 비슷한 부부은 encoder 부분으로 sequence to sequence에서는 context으로 encoder로 압축하는 반면 
Autoencoder는 encoder를 벡터 Z으로 압축한다.   

여기에서 조금 다른 부분은 decoder로 sequence to sequence에서는 번역할 sentence가 token의 형태로 들어가는 것이고, 
Autoencoder에서는 다시 encoder에 들어갔던 값들을 다시 output에 넣어서 비교한다. 즉 입력과 출력이 같은 값으로 하는 
것이다.   

Autoencoder에 대하여 덧붙인다면 input과 output을 같은 값으로 훈련시키므로, 이는 anomaly detection 유용했다.
특히나 outlier 잡아내는데 썼다.   

sequence to sequence로 돌아오면, encoder에 입력할 문장을 넣은 후 RNN에 들어가 연산 후에 RNN의 hidden vector를
decoder에 넘겨 주어 출력할 문장을 넣고, 훈련시킨다. 단, 출력할 문장은 decoder 입력 문장과 출력 문장이 무엇이 다른 지를
잘 보면 입력 문장 앞에는 &lt;sos&gt;, 출력 문장 뒤에는 &lt;eos&gt;가 다른 것 외는 모두 같다.   
- &lt;sos&gt;: start of sentence의 약자로 문장이 시작한다는 것을 인공지능에게 알리는 토큰 
- &lt;eos&gt;: end of sentence의 약자로 문장이 끝났다는 것을 인공지능에게 알리는 토큰

그러면 어떻게 훈련데이터를 token으로 만들어 모델이 훈련하기 좋게 만드는지 Dataset의 구현과 DataLoadeer를 먼저 살피고   
공부하면서 만든 sequence to sequence 모델을 보자.   

- Dataset 구현   

```python
class Translation_Dataset(Dataset):

    def __init__(self, data, en_corpus, ger_corpus, spacy_en, spacy_ger):
        super(Translation_Dataset, self).__init__()

        self.dataset = data
        self.en_corpus = en_corpus
        self.ger_corpus = ger_corpus
        self.spacy_en = spacy_en
        self.spacy_ger = spacy_ger


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.tokenizer_eng(self.dataset[:, 0][item]), self.tokenizer_ger(self.dataset[:, 1][item])

    def tokenizer_ger(self, text: str):
        new_string = re.sub(r"[^a-zA-Z0-9 ]", "", text)
        new_string = new_string.lower()
        return [self.ger_corpus['<sos>']] + [self.ger_corpus[word] for word in
                                             [tok.text for tok in self.spacy_ger.tokenizer(new_string)]], \
               [self.ger_corpus[word] for word in [tok.text for tok in self.spacy_ger.tokenizer(new_string)]] + 
                [self.ger_corpus['<eos>']]
        # return new_string

    def tokenizer_eng(self, text: str):
        new_string = re.sub(r"[^a-zA-Z0-9 ]", "", text)
        new_string = new_string.lower()
        return [self.en_corpus[word] for word in [tok.text for tok in self.spacy_en.tokenizer(new_string)]] + [
            self.en_corpus['<eos>']]
```   
Dataset에서 먼저 살펴볼 것은 어떻게 영어 문장과 독일어 문장을 token으로 바꾸어서 출력하는지를 보았으면 한다.   
다른 블로그를 찾아 보니 bucketiterator가 deprecated 되었는데도 이것으로 짜여진 코드가 돌아다녀서 답답했는데 구현해 낼 수 있었다.   
tokenizer_ger 와 tokenizer_eng 처음의 공통된 과정으로 모두 문장을 소문자로 바꾼다.   
다른점은 영어에서 독일어로 바꾸는 번역 모델을 만들 것이기 때문에 영어는 뒤에 &lt;eos&gt;를 붙여서 내어 보내지만 독일어의 경우는 다르다.   
독일어는 같은 문장을 내어 보내지만 하나는 &lt;sos&gt;를 맨 앞에 붙이고 다른 하나는 &lt;eos&gt;를 붙여서 내어 보낸다. 
- DataLoader 구현   

```python
def collate_fn(batch_size):
    input_en_list, input_ger_list, ouput_ger_list = [], [], []

    for input_en, (input_ger, output_ger) in batch_size:
        input_en = torch.tensor(input_en)
        input_en_list.append(input_en)

        input_ger = torch.tensor(input_ger)
        input_ger_list.append(input_ger)

        output_ger = torch.tensor(output_ger)
        ouput_ger_list.append(output_ger)

    input_en_tensors = pad_sequence(input_en_list, padding_value=en_vocab['<pad>'])
    input_ger_tensors = pad_sequence(input_ger_list, padding_value=ger_vocab['<pad>'])
    output_ger_tensors = pad_sequence(ouput_ger_list, padding_value=ger_vocab['<pad>'])

    return input_en_tensors, input_ger_tensors, output_ger_tensors


sample_set = Translation_Dataset(dataset, en_vocab, ger_vocab, spacy_en, spacy_ger)
sampleLoader = DataLoader(sample_set, batch_size=3, shuffle=True, collate_fn=collate_fn)
```
DataLoader는 기존의 pytorch의 문법과 크게 다를 것이 없다.   
다른 점은 문장마다 길이가 다르므로 batch로 묶을 때에 문제가 생긴다. 그럴 때에는 제일 긴 문장을 기준으로 짧은 문장에 &lt;pad&gt;를 붙여 보완한다. 
그것을 처리하기 위하여 **pad_sequence**로 만든 것이 collate_fn 함수이고, 이를 DataLoader에 넣는다. 
- encoder   

````python
class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, pad, p):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=pad)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # print("in Encoder through parameter x's size:\t", x.size())
        embedding = self.embedding(x)
        outputs, (hidden, cell) = self.rnn(embedding)


        return hidden, cell
````   
encoder에 대한 설명은 어느 정도 간단하다.   
encoder에서는 token으로 변환된 문장이 embedding으로 들어가 tensor으로 바뀌는데 (batch size, sequence length, embedding size)로 바뀐다.   
이는 LSTM층으로 들어가 순서 대로 들어 가나, 여기서 결과 값으로 나오는 output을 버리고, LSTM의 hidden state, cell state를 이를 decoder에 넘겨준다.
- decoder   

```python
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, pad, p):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=pad)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)

        embedding = self.embedding(x)

        outputs, (_hidden, _cell) = self.rnn(embedding, (hidden, cell))

        prediction = self.fc(outputs.squeeze(0))
        return prediction, _hidden, _cell
```   
이것을 공부하던 나도 Decoder의 생김새를 보고 이해하기 힘들었다. 이게 무슨 뜻인지 전혀 이해하지 못하였다. 왜냐하면 전체적인 맥락을 이해하지 못하기 때문이었다.   
그래서 Decoder는 간단하게만 설명하고 넘어가겠다.   
x로 입력되는 번역할 언어의 토큰 하나가 들어온다. 뭔가 이상하다고? 잠시 뒤에 큰 그림을 볼 때 설명해 주겠다. 토큰은 batch size로 묶인 하나가 들어오게 된다. 
그러므로 이를 unsqueeze를 하여서 tensor 형태를 (1, batch_size, token 1개) 으로 바꾸어 준다.   
그리고 이를 embedding에 넣어서 tensor로 만들고, (1, batch size, embedding size) 으로 바뀌어서 lstm층으로 넘겨준다. 여기서 encoder와 다른 점이 있다.   
encoder에서 넘어온 hidden state와 cell state를 decoder의 rnn에 입력해 주어야 한다. 이것이 저 위에서 본 context vector로 이는 encoder에서 훈련된 정보의 요약판이라 이해하면 된다.   
rnn에서 계산되어 넘어온 outputs는 이를 fc에 넣어서 prediction과 rnn에서 나온 hidden state와 cell state를 반환한다. 왜 decoder에서도 왜 hidden state와 cell state를 반환하냐고? 
다음의 큰 그림을 보면 이해가 갈 것이다. 
- encoder + decoder   

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert self.encoder.hidden_size == self.decoder.hidden_size

        assert self.encoder.num_layers == self.decoder.num_layers

    def forward(self, src, trg, device, teacher_forcing_ratio=0.5):

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.input_size

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)

        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            # print(f" at {t} circle, input state is {input}")
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[t] if teacher_force else top1

        return outputs
```
 드디어 큰 그림으로 보는 sequence to sequence이다. decoder에서 가진 의문을 여기서 풀어 보도록 하겠다.   
 여기서 눈에 들어오는 것은 zero tensor를 만들고 있는 부분일 것이다. 그것도 (target length, batch size, target vocab size) 만큼 만들어 놓는다.   
 궁금증은 다음으로 돌리고 encoder에 번역할 문장 sequence를 넣고 여기서 LSTM의 hidden state, cell state를 받아 온다.   
 그리고 번역된 문장(target)의 맨 앞의 부분의 &lt;sos&gt; 토큰을 꺼내 온다. 그리고 여기서 decoder와 zero tensor가 외 필요한지 다음의 for loop를 보면 이해가 된다.   
   
 for loop는 번역 문장의 길이만큼 돌린다. 여기서 decoder가 등장한다.   
 첫번째 loop가 돌 때에는 decoder에 &lt;sos&gt; 토큰이 decoder에 들어 가게 된다. 물론 encoder에서 가지고 온 hidden state와 cell state와 같이.  
decoder에서 나온 output은 조금 전에 만든 zero tensor, 즉 outputs에 넣자.  