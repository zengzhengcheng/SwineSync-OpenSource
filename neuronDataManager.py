from neuronDataClass import NeuronDataClass
import threading

class NeuronDataManager:
    def __init__(self, providers,batchSize=2,cacheSize=1):

        self.providers = providers
        self.batchSize = batchSize
        if(len(self.providers)<batchSize):
            self.batchSize = len(self.providers)
        self.provider_index = 0
        self.dataClassBatch = None
        self.huancunDataClassBatch=None
        self.huancunProviderIndex=0
        self.huancunBatchSize=cacheSize
        self.cacheSize=cacheSize

    def _wait_provider_ready(self, index):
        """等待指定索引的 provider 完成读取。
        readData() 内部有 try/finally 保证 ready_event 一定会被 set，所以这里无需超时。
        """
        provider = self.providers[index]
        provider.ready_event.wait()  # 无超时：线程正常运行时间可能很长
        if getattr(provider, 'read_error', None) is not None:
            print(f"[ERROR] provider {index} ({provider.filepath}) 读取失败: {provider.read_error}，跳过该文件")
            return False
        return True

    def getInfo(self):
        i=f"已完成文件{self.provider_index-self.batchSize}/{len(self.providers)}\n"
        for index in range(self.provider_index-self.batchSize):
            provider=self.providers[index]
            i+=f"已完成{provider.filepath}\n"
        for key in self.dataClassBatch.keys():
            provider=self.providers[key]
            i+=provider.getState()
        return i

    def get_next_data(self):
        """
        一次性获取多个类实例的数据。

        Args:
            batch_size: 需要获取的类实例数量。

        Returns:
            一个字典，其中键是类实例的名称，值是数据。
        """
        threads = []

        if(self.dataClassBatch is None):
            self.dataClassBatch={}
            self.huancunDataClassBatch={}
            for batch in range(self.batchSize):
                if(self.provider_index<len(self.providers)):
                    self.dataClassBatch[self.provider_index]=self.providers[self.provider_index]
                    assert isinstance(self.dataClassBatch[self.provider_index],NeuronDataClass)
                    thread = threading.Thread(
                        target=self.providers[self.provider_index].readData,
                        daemon=True  # 主程序退出时不被此线程阻塞
                    )
                    threads.append((thread, self.provider_index))
                    thread.start()
                    self.provider_index+=1
                else:
                    break
            for batch in range(self.huancunBatchSize):
                if(self.huancunProviderIndex<self.huancunBatchSize):
                    trueIndex=self.provider_index+self.huancunProviderIndex
                    if(trueIndex<len(self.providers)):
                        thread = threading.Thread(
                            target=self.providers[trueIndex].readData,
                            daemon=True
                        )
                        thread.start()
                        self.huancunProviderIndex+=1
                    else:
                        break
            # 等待初始批次所有线程完成
            # readData() 有 try/finally 保证，线程一定会结束，无需超时
            for thread, _ in threads:
                thread.join()

        count = 0
        dellist=[]
        addlist=[]
        dataDict={}
        # 安全计数器：防止 batchSize 与 dataClassBatch 不一致时死循环
        max_iterations = (len(self.providers) + 1) * 2
        iteration = 0
        while count < self.batchSize:
            iteration += 1
            if iteration > max_iterations:
                print(f"[WARN] get_next_data while 循环超过安全上限 {max_iterations}，强制退出")
                break
            if not self.dataClassBatch:
                break
            for key, current_provider in list(self.dataClassBatch.items()):
                current_provider = self.providers[key]
                data = current_provider.getData()
                if data is not None:
                    dataDict[key] = data
                    count += 1
                else:
                    current_provider.saveData()
                    dellist.append(key)
                    # 当前类实例结束，切换到下一个
                    if (self.provider_index < len(self.providers)):
                        addlist.append(self.provider_index)
                        if(self.huancunProviderIndex==0):
                            self.providers[self.provider_index].readData()
                            dataDict[self.provider_index] = self.providers[self.provider_index].getData()
                            self.provider_index += 1
                        else:
                            # 等待预读线程，带超时
                            ok = self._wait_provider_ready(self.provider_index)
                            self.huancunProviderIndex -= 1
                            if ok:
                                dataDict[self.provider_index] = self.providers[self.provider_index].getData()
                            self.provider_index += 1
                    count += 1

        for key in dellist:
            delclass = self.dataClassBatch.pop(key, None)
            del delclass
        for key in addlist:
            self.dataClassBatch[key] = self.providers[key]
            trueIndex = self.provider_index + self.huancunProviderIndex
            if (trueIndex < len(self.providers)):
                thread = threading.Thread(
                    target=self.providers[trueIndex].readData,
                    daemon=True
                )
                thread.start()
                self.huancunProviderIndex += 1
        self.batchSize=len(self.dataClassBatch)
        if not dataDict:
            return None  # 所有类实例都已结束
        return dataDict

    def send_output(self, output):
        for key in output.keys():
            provider=self.providers[key]
            start,labels,cleanecg =output[key]
            provider.setlabel(start,labels,cleanecg)

if __name__=="__main__":
    # 使用示例
    filepaths=[r"C:\Users\zengz\Desktop\7个整G原始数据(108639026)\ECGLog-2024-5-24-13-12-19.txt"]
    savepath=r"C:\Users\zengz\Desktop\7个整G原始数据(108639026)"
    providers = [NeuronDataClass(filepath, savepath) for filepath in filepaths]
    neuronDataManager = NeuronDataManager(providers, batchSize=4)
    detector = NeuronDataManager(None,5120*3)
    detector.get_next_data()
