require('mobdebug').start()

require 'experience_database'

db = torch.load("test.db")

c = db:get_mini_batch(5,true)

print(c)


