import sys

f = open(sys.argv[1], "r")

lines = f.readlines()

io  = 0.0
sum = 0.0
n   = 0

for line in lines:
    if line.count("Timing for main:"):
        n += 1
        time =  float(line.split()[8])
        if time < 100.0:
           sum = sum + time
        else:
           io = io + time
           print line, time

print n
print io
print sum
