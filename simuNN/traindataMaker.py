'''make example for machine learning
'''

import numpy as np
from dataDim import *

def makeHeiWeiData(seed ,size) :
    rdm = np.random.RandomState( seed )
    X = rdm.rand(size ,2 )
    X *= 4
    return X    

'''[ height/std_height , weight/std_weight]  => [ healthy (1.0) or no(0.0)]'''
def  makeHeiWeiTrainData(x , y , mode ,seed , size  ):
    heiwei =  makeHeiWeiData(seed,size)   
    wh = heiwei[:,1] /heiwei[:,0]
    wh1 = wh <  1
    ntwh0 = wh1.astype ( np.float32  )
    ntwh1 = np.reshape( ntwh0, ( len(ntwh0) ,1 ) )
    with open( 'trainData.py' , mode ) as f:
        if mode == 'wt' :
            str0 = "'''\n"
            f.write( str0  )
            str0 = x+"=[ height/std_height , weight/std_weight]\n"
            f.write( str0  )
            str0 = y+"=[ healthy or not ]\n'''\n"
            f.write( str0  )
            str0  = 'TRAIN_THRESHOLD = 0.3 \n'
            f.write( str0  ) 
        str0 = x+'=[' 
        f.write( str0  )
        for i in range( len( heiwei )  ) : 
            str0 = np.array2string ( heiwei[i]  , separator = ',' )
            if  i <  len( heiwei ) -1 :
                str0 += ','
            str0 += '\n'
            f.write( str0)
        str0 = ']\n'
        f.write( str0)
        str0 = y+'=['
        f.write( str0  )
        for i in range( len(ntwh1) ): 
            str0 = np.array2string ( ntwh1[i], separator = ',' )
            if i <  len(ntwh1) -1:
                str0 += ','
            str0 += '\n'
            f.write( str0  )
        str0 = ']\n'
        f.write( str0 )
        sum1 = ntwh1.sum()
        print(sum1)
        str0 = 'ONE_NUM='  + str( sum1 ) + '\n'
        f.write( str0 )

def  heiWeiTrainMakerMain() :
    makeHeiWeiTrainData( 'X' , 'Y_' , 'wt' , 12  ,   SAMPLE_SIZE )
    makeHeiWeiTrainData( 'XT' , 'Y_T' , 'at'  , 1001 , VALIDATE_SIZE )

''' using vocabulary and reading to predict English Score
   vocabulary is sampled  using  thousand words grasped as unit
   reading is sample using thousand words read as unit
   Score is sampled  at  Scope [0.0~5.0]
'''
def makeVocabulReadingData(seed ,size) :
    rdm = np.random.RandomState( seed )
    X = rdm.rand(size ,2 )
    X[:,0] *= 8
    X[:,1] *= 10
    return X    

'''[ Vocabulary/1000 , Reading/1000000]  => [ English Score between  0 and 5]
    Vocabulary is about 1000 to 10000 , normal is 3000 , so Voc normal is 3
   400 wors per page , 200 page per books , 20 textbooks +10  other books   is a normal
   reading,  so reading normal  is 400*200*30/1000000=2.4
   Y(Real Score ) = 0.65*x1(Vocabulary) + 0.1*x2(Reading)
   Change Score to the scope between 0 and 100 , * 20 
'''
def  makeVocabulReadingTrainData(x , y , mode ,seed , size  ):
    vocReading =  makeVocabulReadingData(seed,size)   
    score0 = vocReading[:,0]*0.55 + vocReading[:,1]*0.2
    score0[:]  *= 20 
    #score0[ score0 > 5 ] = 5
    score0[ score0 > 100 ] = 100
    score1 = np.reshape( score0, ( len(score0) ,1 ) )
    with open( 'trainData.py' , mode ) as f:
        if mode == 'wt' :
            str0 = "'''\n"
            f.write( str0  )
            str0 = x+"=[ Vocabulary , Reading]\n"
            f.write( str0  )
            str0 = y+"=[ English score ]\n'''\n"
            f.write( str0  )
            str0  = 'TRAIN_THRESHOLD = 5\n'
            f.write( str0  ) 
        str0 = x+'=[' 
        f.write( str0  )
        for i in range( len( vocReading )  ) : 
            str0 = np.array2string ( vocReading[i]  , separator = ',' )
            if  i <  len( vocReading ) -1 :
                str0 += ','
            str0 += '\n'
            f.write( str0)
        str0 = ']\n'
        f.write( str0)
        str0 = y+'=['
        f.write( str0  )
        for i in range( len(score1) ): 
            str0 = np.array2string ( score1[i], separator = ',' )
            if i <  len(score1) -1:
                str0 += ','
            str0 += '\n'
            f.write( str0  )
        str0 = ']\n'
        f.write( str0 )

def  VocabulReadingTrainMakerMain() :
    makeVocabulReadingTrainData( 'X' , 'Y_' , 'wt' , 12  ,   SAMPLE_SIZE )
    makeVocabulReadingTrainData( 'XT' , 'Y_T' , 'at'  , 1001 , VALIDATE_SIZE )



#heiWeiTrainMakerMain()
VocabulReadingTrainMakerMain()
