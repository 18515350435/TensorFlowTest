import tensorflow as tf

# 创建两个矩阵
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
# 创建一个矩阵乘法的操作节点
product = tf.matmul(m1,m2)

#创建一个会话
sess = tf.Session()
# 执行会话
result = sess.run(product)
sess.close()
print(result)

