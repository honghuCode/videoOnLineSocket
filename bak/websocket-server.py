
import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.python import log
from twisted.internet import reactor

import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random



class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol(WebSocketServerProtocol):
   
    def __init__(self):
        self.images = {}
        self.training = True
        self.people = []
        self.svm = None
        self.count = 0

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        rs = self.processFrame(msg)
        print(rs)
        if(rs == -1):
            self.sendMessage('{"type": "FINISH"}')
        else:
            self.sendMessage('{"type": "PROCESSED"}')
        #print(msg['data'])
        '''
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "ALL_STATE":
            self.loadState(msg['images'], msg['training'], msg['people'])
        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            self.processFrame(msg['dataURL'], msg['identity'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            if not self.training:
                self.trainSVM()
        elif msg['type'] == "ADD_PERSON":
            self.people.append(msg['val'].encode('ascii', 'ignore'))
            print(self.people)
        elif msg['type'] == "UPDATE_IDENTITY":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                self.images[h].identity = msg['idx']
                if not self.training:
                    self.trainSVM()
            else:
                print("Image not found.")
        elif msg['type'] == "REMOVE_IMAGE":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                del self.images[h]
                if not self.training:
                    self.trainSVM()
            else:
                print("Image not found.")
        elif msg['type'] == 'REQ_TSNE':
            self.sendTSNE(msg['people'])
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))
        '''
    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople):
        self.training = training

        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')
            self.images[h] = Face(np.array(jsImage['representation']),
                                  jsImage['identity'])

        for jsPerson in jsPeople:
            self.people.append(jsPerson.encode('ascii', 'ignore'))

        if not training:
            self.trainSVM()

    def getData(self):
        X = []
        y = []
        for img in self.images.values():
            X.append(img.rep)
            y.append(img.identity)

        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
            return None

        if args.unknown:
            numUnknown = y.count(-1)
            numIdentified = len(y) - numUnknown
            numUnknownAdd = (numIdentified / numIdentities) - numUnknown
            if numUnknownAdd > 0:
                print("+ Augmenting with {} unknown images.".format(numUnknownAdd))
                for rep in self.unknownImgs[:numUnknownAdd]:
                    # print(rep)
                    X.append(rep)
                    y.append(-1)

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

  



    def processFrame(self,  msg):
        firstFrame = ''
        if(msg['data'] != 'NULL'):
            head = "data:image/jpeg;base64,"
            assert(msg['data'].startswith(head))
            imgdata = base64.b64decode(msg['data'][len(head):])
            imgF = StringIO.StringIO()
            imgF.write(imgdata)
            imgF.seek(0)
            img = Image.open(imgF)

            buf = np.fliplr(np.asarray(img))
            rgbFrame = np.zeros((360, 640, 3), dtype=np.uint8)
            rgbFrame[:, :, 0] = buf[:, :, 2]
            rgbFrame[:, :, 1] = buf[:, :, 1]
            rgbFrame[:, :, 2] = buf[:, :, 0]
            name = int(random.random() * 1000)
            self.count = self.count + 1
            if(self.count == 1):#first frame
                firstFrame = msg['data']
                cv2.imwrite('./frame/'+str(self.count)+'.jpg', rgbFrame)
                return 1
            else:
                if(msg['data'] == firstFrame):
                    return -1
                else:
                    cv2.imwrite('./frame/'+str(self.count)+'.jpg', rgbFrame)
                    return 1
        else:
            return 0    

        # cv2.imshow('frame', rgbFrame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return
if __name__ == '__main__':
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:9000")
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(9000, factory)
    reactor.run()
