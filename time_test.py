import time
import datetime

# 현재 timestamp
print(time.time())

# 현재 시간 (readable format)
print(datetime.datetime.now())

# 시스템 시간과 UTC 시간 비교
print("System time:", time.strftime('%Y-%m-%d %H:%M:%S'))
print("UTC time:", datetime.datetime.utcnow())