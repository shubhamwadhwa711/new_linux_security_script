import random
from datetime import datetime
now = datetime.now()
num = random.randint(1,100000)

with open('rend.text','a') as f:
    f.write('{} -your random numer is {}\n'.format(now,num))




# from crontab import CronTab

# jobScheduler = CronTab(user='root')

# job = jobScheduler.new(command='intro_full_text_metadata.py')

# job.hour.every(1)
# jobScheduler.write()