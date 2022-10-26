def getNextPowerOf2(x):
  return 1<<(x-1).bit_length()

def getBeforePowerOf2(x):
  return 1<<(x-1).bit_length()-1