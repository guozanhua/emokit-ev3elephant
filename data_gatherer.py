__author__ = 'igor sieradzki'

from emokit.emotiv import Emotiv
import gevent
from datetime import datetime
from time import time, sleep
import os


BUFFER_CAPACITY = 100000000000
TEST = False

def console_gather() :

    os.system('clear')

    headset = Emotiv()
    gevent.spawn(headset.setup)
    gevent.sleep(0)

    packets = 0
    now = datetime.now()
    filename = str(now.time()) + "_" + str(now.date())
    while True :
        dir = str(input("Choose input: \n 1. up\n 2. down\n 3. left\n 4. right\n 0. neutral\n"))
        if dir in ['1','2','3','4','0'] : break
    filename += "_" + dir
    if TEST : filename = "TEST_" + filename


    buffers = []
    names = 'AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4'.split(' ')
    for name in names :
        buffers.append(sensor_buffer(name))

    print "Training will start in..."; sleep(1); print "3..."; sleep(1); print "2..."; sleep(1); print "1..."; sleep(1); print "Focus!"

    qualities = []
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

    quality = 0.
    f = open("./data/" + filename,'w')
    columns = []
    for name in names :
        columns.append(str(name))
        columns.append('Q' + str(name))
    f.write(','.join(columns))
    f.write('\n')
    while packets > 0 :
        for buffer in buffers :
            f.write( buffer.pop() )
        f.write('\n')
        packets -= 1

    f.close()


    print "Finished reading, saved to file %s" % filename
    for buffer in buffers :
        print "Sensor %s mean quality: %.2f" % (buffer.name, buffer.mean_quality())

def class_gather(label, headset) :

    packets = 0
    now = datetime.now()
    filename = str(now.time()) + "_" + str(now.date())

    filename += "_" + label

    buffers = []
    names = 'AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4'.split(' ')
    for name in names :
        buffers.append(sensor_buffer(name))


    if label == '0' : print "> Neutral position"
    elif label == '1' : print "> Forward Movement (UP)"
    elif label == '2' : print "> Backward Movement (DOWN)"
    elif label == '3' : print "> Left Steering (LEFT)"
    elif label == '4' : print "> Right Steering (RIGHT)"

    print "Training will start in..."; sleep(1); print "3..."; sleep(1); print "2..."; sleep(1); print "1..."; sleep(1); print "Focus!"

    qualities = []
    timeout = time() + 12
    while True:
        if time() > timeout :
            break

        packet = headset.dequeue()
        for buffer in buffers :
            buffer.update( packet )
        packets += 1
        gevent.sleep(0)

    f = open("./data/" + filename,'w')
    columns = []
    for name in names :
        columns.append(str(name))
        columns.append('Q' + str(name))
    f.write(','.join(columns))
    f.write('\n')
    while packets > 0 :
        for buffer in buffers :
            f.write( buffer.pop() )
        f.write('\n')
        packets -= 1

    f.close()

def session() :

    headset = Emotiv()
    gevent.spawn(headset.setup)
    gevent.sleep(0)

    labels = ['0', '1', '2', '3', '4']

    for label in labels :
        os.system('clear')
        print "Training for class: ", label
        raw_input("Press Enter to start training..." )
        class_gather(label, headset)

    headset.close()

class sensor_buffer :

    def __init__(self, name) :

        self.name = name
        self.buffer = []
        self.sum_quality = 0.
        self.n_readings = 0

    def update(self, packet) :

        if len(self.buffer) > BUFFER_CAPACITY :
            raise Exception('Buffer overflow')
        self.buffer.append( [packet.sensors[self.name]['value'], packet.sensors[self.name]['quality']] )
        self.sum_quality += int(packet.sensors[self.name]['quality'])
        self.n_readings += 1

    def pop(self) :
        # if len( self.buffer > 0 ) :
        data = self.buffer.pop()
        return str( reading(data) )

    def mean_quality(self) :
        return self.sum_quality / self.n_readings

class reading :

    def __init__(self, data):

        self.value = data[0]
        self.quality = data[1]

    def __repr__(self) :
        return "%i,%.1f," % ( self.value, self.quality )
