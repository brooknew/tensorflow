import tensorflow as tf
checkPATH =  'model/' #save的目录名
checkFILE = 'saverTest' #save 数据文件的前缀

def saveVariables() :
    print( "start save" ) 
    a = tf.Variable( [12,102,10002] ) #第一个需要restore的变量
    b = tf.Variable(2*10)
    #a  = tf.multiply(a , 2  ) 
    sav = tf.train.Saver( )
    se = tf .Session()
    init =tf.global_variables_initializer()
    se.run( init )
    for i in range( 3 ) :
        r = se.run( a )
        a = a*2
        print( r )
        sav.save(  se ,  checkPATH + checkFILE , global_step = i)
    
    #sav.restore(  se ,  'model/')
    se.close()
    

def restoreVariables() :
    print("restore strat" ) 
    va1 = tf.Variable( [0,0,00] )#第一个需要restore的变量，
                                             #名字可以与save时的不一样，但是类型要一样
    #b = tf.Variable(0) 
    sav = tf.train.Saver(  )
    se = tf .Session()
    init =tf.global_variables_initializer()
    se.run( init )
    ckpt = tf.train.get_checkpoint_state( checkPATH )
    print( "ckpt:" ,  ckpt )
    print (" path:" ,ckpt.model_checkpoint_path ) 
    if ckpt and ckpt.model_checkpoint_path:
        sav.restore(se , ckpt.model_checkpoint_path)
               #ckpt.model_checkpoint_path 是最新的存储文件名
    r = se.run( va1 )
    print( r )
    #print( se.run(b) )
    se.close() 

#save 和 restore 要分开不同的应用执行。在同一个应用里先后串行执行，restore时会失败   

#saveVariables() 
restoreVariables()
