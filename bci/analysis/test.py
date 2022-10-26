def findNextPowerOf2(n):
  n = n - 1
  while n & n - 1:
      n = n & n - 1
  return n << 1

def findNextPowerOf2(x):
  return 1<<(x-1).bit_length()

def findBeforePowerOf2(x):
  return 1<<(x-1).bit_length()-1

if __name__ == '__main__':
  n = 38
  print(findNextPowerOf2(n))