import numpy as np
import pandas as pd

def cengci(filename):#输入：矩阵存储文件名，输出：矩阵权重分配和CR值
    matrix=[]
    with open(filename,'r') as file:
        lines=file.readlines()
        dimension=len(lines)#矩阵维度
        for line in lines:
            row=line.strip().split()
            processed_row = []
            for element in row:
                if '/' in element:  # 判断是否为分数形式，如果是保留两位小数
                    numerator, denominator = element.split('/')
                    processed_element = round(float(numerator) / float(denominator), 2)
                else:
                    processed_element = float(element)
                processed_row.append(processed_element)
            matrix.append(processed_row)#从文件中读取矩阵
    ri=[0,0,0.58,0.9,1.12,1.24,1.32,1.41,1.45,1.49]#一致性指标
    eigenvalues,eigenvectors=np.linalg.eig(matrix)
    D=np.diag(eigenvalues)
    V=eigenvectors
    lamda=np.max(D)
    indices=np.where(D==lamda)
    row_index,col_index=indices[0][0],indices[1][0]
    column=V[:,col_index]
    w0=column/np.sum(column)#计算权重
    cr0=(lamda-dimension)/(dimension-1)/ri[dimension]#计算CR值
    return w0,cr0
def SQ(data):
    rows=data.shape[0]
    cols=data.shape[1]#输入矩阵的大小，rows为行数（对象个数），cols为列数（指标个数）
    R=data
    Rmin=np.min(R,axis=0)#矩阵中最小行
    Rmax=np.max(R,axis=0)#矩阵中最大行
    A=Rmax-Rmin#分母 矩阵中最大行减最小行
    y=R-np.tile(Rmin,(rows,1))#分子 R矩阵每一行减去最小行
    for j in range(cols):#该循环用于正向指标标准化处理 分子/分母
        y[:,j]=y[:,j]/A[j]
    S=np.sum(y,axis=0)#列之和（用于列归一化）
    Y=np.zeros((rows,cols))
    for i in range(cols):
        Y[:,i]=y[:,i]/S[i]#该循环用于列的归一化
    k=1/np.log(rows)
    lnYij1=np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if Y[i,j]==0:
                lnYij1[i,j]=0
            else:
                lnYij1[i,j]=np.log(Y[i,j])#循环遍历取对数
    ej1=-k*np.sum(Y*lnYij1,axis=0)#计算正向指标标准化熵值ej1
    weights1=(1-ej1)/(cols-np.sum(ej1))#正向指标权重weights1
    return weights1
def main():
    matfirstlayer='cengci1.txt'
    w0,cr0=cengci(matfirstlayer)#计算第一层权重和CR值
    matx='cengci2_1.txt'
    w2_1,cr2_1=cengci(matx)#计算第二层中X部分的权重和CR值
    matim='cengci2_2.txt'
    w2_2,cr2_2=cengci(matim)#计算第二层中IM部分的权重和CR值
    matic='cengci2_3.txt'
    w2_3,cr2_3=cengci(matic)#计算第二层中IC部分的权重和CR值
    matar='cengci2_4.txt'
    w2_4,cr2_4=cengci(matar)#计算第二层中AR部分的权重和CR值
    matland='cengci2_5.txt'
    w2_5,cr2_5=cengci(matland)#计算第二层中着舰部分的权重和CR值
    cengci_result=[]
    cr_result=[cr0]
    for i in range(len(w0)):
        name='w2_'+str(i+1)
        value=eval(name)
        result=w0[i]*value
        cengci_result=np.append(cengci_result,result)#计算层次分析法权重结果
    for i in range(len(w0)):
        name='cr2_'+str(i+1)
        value=eval(name)
        cr_result=np.append(cr_result,value)#计算层次分析法CR值结果
    trainfile='train_data.csv'#训练数据
    data_train=pd.read_csv(trainfile)
    datatrain=data_train.values
    row=datatrain.shape[0]
    column=datatrain.shape[1]
    datatrainabs=np.abs(datatrain)#绝对值
    ax_x=0#每个指标的正态隶属度归一化函数中有两个参数，定为a和b；ax_x中前一个x代表x方向偏差，后一个x代表X位置；X位置x方向偏差的正态隶属度函数中a参数
    bx_x=1.5#X位置x方向偏差的正态隶属度函数中b参数
    ay_x=0#X位置y方向偏差的正态隶属度函数中a参数
    by_x=1#X位置y方向偏差的正态隶属度函数中b参数
    az_x=1#X位置z方向偏差的正态隶属度函数中a参数
    bz_x=1#X位置z方向偏差的正态隶属度函数中b参数
    ax_im=0#ax_im指的是IM位置x方向偏差的正态隶属度函数中a参数
    bx_im=1.5#IM位置x方向偏差的正态隶属度函数中b参数
    ay_im=0#IM位置y方向偏差的正态隶属度函数中a参数
    by_im=1#IM位置y方向偏差的正态隶属度函数中b参数
    az_im=0#IM位置z方向偏差的正态隶属度函数中a参数
    bz_im=1#IM位置z方向偏差的正态隶属度函数中b参数
    ax_ic=0#IC位置x方向偏差的正态隶属度函数中a参数
    bx_ic=1.5#IC位置x方向偏差的正态隶属度函数中b参数
    ay_ic=0#IC位置y方向偏差的正态隶属度函数中a参数
    by_ic=1#IC位置y方向偏差的正态隶属度函数中b参数
    az_ic=0#IC位置z方向偏差的正态隶属度函数中a参数
    bz_ic=1#IC位置z方向偏差的正态隶属度函数中b参数
    ax_ar=0#AR位置x方向偏差的正态隶属度函数中a参数
    bx_ar=1.5#AR位置x方向偏差的正态隶属度函数中b参数
    ay_ar=0#AR位置y方向偏差的正态隶属度函数中a参数
    by_ar=1#AR位置y方向偏差的正态隶属度函数中b参数
    az_ar=0#AR位置z方向偏差的正态隶属度函数中a参数
    bz_ar=1#AR位置z方向偏差的正态隶属度函数中b参数
    avside_ar=0#AR位置侧向速度偏差的正态隶属度函数中a参数
    bvside_ar=1#AR位置侧向速度偏差的正态隶属度函数中b参数
    avdown_ar=0#AR位置下沉速度偏差的正态隶属度函数中a参数
    bvdown_ar=1#AR位置下沉速度偏差的正态隶属度函数中b参数
    aphi_ar=0#AR位置滚转角偏差的正态隶属度函数中a参数
    bphi_ar=2#AR位置滚转角偏差的正态隶属度函数中b参数
    atheta_ar=0#AR位置俯仰角偏差的正态隶属度函数中a参数
    btheta_ar=1.5#AR位置俯仰角偏差的正态隶属度函数中b参数
    avdown_fn=0#着舰位置下沉速度偏差的正态隶属度函数中a参数
    bvdown_fn=1#着舰位置下沉速度偏差的正态隶属度函数中b参数
    azend_fn=0#着舰位置舰尾净高偏差的正态隶属度函数中a参数
    bzend_fn=1#着舰位置舰尾净高偏差的正态隶属度函数中b参数
    atheta_fn=0#着舰位置俯仰角偏差的正态隶属度函数中a参数
    btheta_fn=1#着舰位置俯仰角偏差的正态隶属度函数中b参数
    aphi_fn=0#着舰位置滚转角偏差的正态隶属度函数中a参数
    bphi_fn=1#着舰位置滚转角偏差的正态隶属度函数中b参数
    apsi_fn=0#着舰位置偏航角偏差的正态隶属度函数中a参数
    bpsi_fn=1#着舰位置偏航角偏差的正态隶属度函数中b参数
    a=np.array([ay_x,az_x,ay_im,az_im,ay_ic,az_ic,ay_ar,az_ar,avside_ar,avdown_ar,aphi_ar,atheta_ar,avdown_fn,azend_fn,atheta_fn,aphi_fn,apsi_fn])
    b=np.array([by_x,bz_x,by_im,bz_im,by_ic,bz_ic,by_ar,bz_ar,bvside_ar,bvdown_ar,bphi_ar,btheta_ar,bvdown_fn,bzend_fn,btheta_fn,bphi_fn,bpsi_fn])
    #矩阵运算
    miux = np.zeros((row, 1))
    for j in range(row):
        if datatrainabs[j, 17] < 0 and datatrainabs[j, 17] >= -6.1:
            miux[j, 0] = (datatrainabs[j, 17] + 6.1) / 6.1
        elif datatrainabs[j, 17] >= 0 and datatrainabs[j, 17] <= 6.1:
            miux[j, 0] = (datatrainabs[j, 17] - 6.1) / -6.1
        else:
            miux[j, 0] = 0#落点纵向偏差的三角隶属度归一化
    miuy = np.zeros((row, 1))
    for j in range(row):
        if datatrainabs[j, 18] < 0 and datatrainabs[j, 18] >= -1.52:
            miuy[j, 0] = (datatrainabs[j, 18] + 1.52) / 1.52
        elif datatrainabs[j, 18] >= 0 and datatrainabs[j, 18] <= 1.52:
            miuy[j, 0] = (datatrainabs[j, 18] - 1.52) / -1.52
        else:
            miuy[j, 0] = 0#落点横向偏差的三角隶属度归一化
    
    d1=datatrainabs[:,:17]#取训练数据前17列（指定变量顺序），第18列是落点纵向偏差，第19列是落点横向偏差
    d2=1-np.exp(-((d1-a)/b)**2)#除落点纵向偏差和落点横向偏差外的参数用正态隶属度函数归一化
    d3 = np.hstack((d2[:, :17], 1 - miux, 1 - miuy))
    datatrain1 = d3
    shangquan_result=SQ(datatrain1)#熵权法计算权重
    sum3=0
    quanzhong=[]
    for i in range(len(cengci_result)):
        sum3=sum3+cengci_result[i]*shangquan_result[i]#参数层次权重*熵权权重再求和
    for i in range(len(cengci_result)):
        quanzhong.append(cengci_result[i]*shangquan_result[i]/sum3)#合成归一法求最终权重，参数层次权重*熵权权重/总和
    testfile='test_data.csv'#测试数据
    data_test=pd.read_csv(testfile)
    datatest=data_test.values
    rowtest=datatest.shape[0]
    coltest=datatest.shape[1]
    datatestabs=np.abs(datatest)
    miuxt = np.zeros((rowtest, 1))
    for j in range(rowtest):
        if datatestabs[j, 17] < 0 and datatestabs[j, 17] >= -6.1:
            miuxt[j, 0] = (datatestabs[j, 17] + 6.1) / 6.1
        elif datatestabs[j, 17] >= 0 and datatestabs[j, 17] <= 6.1:
            miuxt[j, 0] = (datatestabs[j, 17] - 6.1) / -6.1
        else:
            miuxt[j, 0] = 0#三角隶属度归一化
    miuyt = np.zeros((rowtest, 1))
    for j in range(rowtest):
        if datatestabs[j, 18] < 0 and datatestabs[j, 18] >= -1.52:
            miuyt[j, 0] = (datatestabs[j, 18] + 1.52) / 1.52
        elif datatestabs[j, 18] >= 0 and datatestabs[j, 18] <= 1.52:
            miuyt[j, 0] = (datatestabs[j, 18] - 1.52) / -1.52
        else:
            miuyt[j, 0] = 0#三角隶属度归一化
    d1t=datatestabs[:,:17]
    d2t=1-np.exp(-((d1t-a)/b)**2)#正态隶属度归一化
    d3t = np.hstack((d2t[:, :17], 1 - miuxt, 1 - miuyt))
    datatest1 = d3t
    result_zuhefuquan=np.zeros((rowtest,1))
    for i in range(rowtest):
        for j in range(coltest):
            result_zuhefuquan[i]=datatest1[i,j]*quanzhong[j]+result_zuhefuquan[i]
    print(result_zuhefuquan)#测试数据的评估结果
if __name__ == '__main__':
    main()