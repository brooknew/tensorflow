
import tensorflow as tf
import numpy as np 

def saveMatrix( t ,  tname , fname , mode ) :
    with open ( fname ,  mode  ) as file:
        sa = tname +  ' : [\n' 
        file.write( sa  ) 
        for  i in range( len( t ) ) :
            st = np.array2string( t[i]  , separator=',')
            if  i <  len( t ) - 1 :
                st += ','
            st += '\n'
            file.write( st ) 
        file.write( ']\n' )


a =np.array( [[1,2],
       [3,4]
     ] )
b = np.array( [11,21,31] )

saveMatrix( a , 'a' , 'ab.txt' , 'wt')
saveMatrix( b , 'b' , 'ab.txt' , 'at')
