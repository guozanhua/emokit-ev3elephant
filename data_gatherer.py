__author__ = 'igor sieradzki'

from emokit.emotiv import Emotiv
import gevent
from datetime import datetime
from time import time, sleep


BUFFER_CAPACITY = 100000000000;
TEST = True

def gather() :

    headset = Emotiv()
    gevent.spawn(headset.setup)
    gevent.sleep(0)


    packets = 0
    now = datetime.now()
    filename = str(now.time()) + "_" + str(now.date())
    while True :
        dir = str(input("Choose input: \n 1. up\n 2. down\n 3. left\n 4. right\n"))
        if dir in ['1','2','3','4'] : break
    filename += "_" + dir
    if TEST : filename = "TEST_" + filename


    buffers = []
    names = 'AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4'.split(' ')
    for name in names :
        buffers.append(sensor_buffer(name))

    print "Training will start in..."; sleep(1); print "3..."; sleep(1); print "2..."; sleep(1); print "1..."; sleep(1); print "Focus!"

    timeout = time() + 12
    while True:
        if time() > timeout :
            break

        packet = headset.dequeue()
        for buffer in buffers :
            buffer.update( packet )
        packets += 1
        gevent.sleep(0)

    headset.close()

    f = open("./data/" + filename,'w')
    while packets > 0 :
        for buffer in buffers :
            f.write( buffer.pop() )
        f.write('\n')
        packets -= 1

    f.close()

class sensor_buffer :

    def __init__(self, name) :

        self.name = name
        self.buffer = []

    def update(self, packet) :

        if len(self.buffer) > BUFFER_CAPACITY :
            raise Exception('Buffer overflow')
        self.buffer.append( [packet.sensors[self.name]['value'], packet.sensors[self.name]['quality']] )

    def pop(self) :
        # if len( self.buffer > 0 ) :
        data = self.buffer.pop()
        return self.name + ":" + str( reading(data) ) + ";"

class reading :

    def __init__(self, data):

        self.value = data[0]
        self.quality = data[1]

    def __repr__(self) :
        return "%i,%.1f" % ( self.value, self.quality )
