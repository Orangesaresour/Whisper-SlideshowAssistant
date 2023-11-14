import cohere
co = cohere.Client('cTJFP4iCWKl8zsptu0wxNyX69KO42sUkfreZITmd') # This is your trial API key
response = co.generate(
  model='command',
  prompt='Please help me turn this weird semi-broken sentence into a coherent sentence with minimal changes. Just make adjustments to the misspellings or words in places that don\'t make sense contextually. Your output should just be a revised sentence that replaces the sentence I input for you. Do not give me any extraneous output except for the revised sentence. Your sentence is: that dark animl is ',
  max_tokens=300,
  temperature=0.9,
  k=0,
  stop_sequences=[],
  return_likelihoods='NONE')
print('Prediction: {}'.format(response.generations[0].text))