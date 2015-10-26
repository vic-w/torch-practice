require 'torch'
require 'gnuplot'

math.randomseed(1)

sample_length = 100
sample_amount = 10000

X = torch.zeros(sample_length)
Y = torch.zeros(sample_length)

for n=2,sample_amount do
  sample_count = 0

  input = {}
  output = {}

  while sample_count < sample_length do
    t_blank = math.random(1,10)
    t_intense = math.random(1,10)
    for i = 1,t_blank do
      table.insert(input, 0)
      table.insert(output, 0)
    end
    table.insert(input, t_intense)
    table.insert(output, 0)
    for i = 1,t_intense do
      table.insert(input, 0)
      table.insert(output, 1)
    end
    sample_count = sample_count+t_blank+t_intense+1
  end

  input = torch.Tensor(input)
  input = input[{{1, sample_length}}]
  output = torch.Tensor(output)
  output = output[{{1, sample_length}}]
  X = X:cat(input,2)
  Y = Y:cat(output,2)
end

torch.save('data.t7', {X:t(), Y:t()})

--gnuplot.pngfigure('img/timer.png')
--gnuplot.plot({input},{output})
--gnuplot.plotflush()

