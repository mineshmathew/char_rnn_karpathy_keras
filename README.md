A keras implementation of Karpathys char-rnn (https://github.com/karpathy/char-rnn)
Although keras already had a text generation example (https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py) using RNNs, it is trained in a many-to-one fashion, where was the original karpathy implementation was in a many-to-many manner. This implementation follows Karpathys style and uses a time distribute dense layer at the output.

To understand it better. consider this, in the keras text generation example ;

input is "where ar" and target is "e"

in this implementation 

input is "where ar" and target is "here are" ( ie target is a sequence one shifted in position)


I have also added an ipynb notebook, hoping to add more details with some figures when I get some free time
