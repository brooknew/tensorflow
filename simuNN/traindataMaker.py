'''make example for machine learning
  [height, weight] => healthy or no
'''

import numpy as np
from dataDim import *

def makeHeiWeiData(seed ,size) :
    rdm = np.random.RandomState( seed )
    X = rdm.rand(size ,2 )
    X *= 4
    return X    


def  makeTrainData(x , y , mode ,seed , size  ):
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

makeTrainData( 'X' , 'Y_' , 'wt' , 12  ,   SAMPLE_SIZE )
makeTrainData( 'XT' , 'Y_T' , 'at'  , 1001 , VALIDATE_SIZE )

