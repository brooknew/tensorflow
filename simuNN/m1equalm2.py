
def printm1m2():
    print( m1 )
    print( m2 )


m1 = [[1,2,3,4],[21,22,23,24] , [31,32,33,34]]
m2 = [[100,200,300,400] , [101,201,301,401]]
printm1m2()

print("list embeded slice")
m2=m1[1:][0:]
print("m1[1:][0:]" , m2 )
m2=m1[1:][1:]
print("m1[1:][1:]" , m2 ) 
m2=m1[0:][1:]
print("m1[0:][1:]" , m2 ) 

#del m1[2]
m2[0][0] = 81100
m2[1][0] = 11100
printm1m2()

print("list embeded slice")
m1=[1,2,3,4]
m2=[10,20,30]
printm1m2()
m1=m2[1:3] ;
m2[2] = 34
printm1m2()
m1 = m2
m2[1] = 56
printm1m2() 
