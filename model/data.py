import  os
from torch.utils.data import  (DataLoader,Dataset)



class ProteinDataset(Dataset):

    def __init__(self,file_path):
        super().__init__()
        self.protains,self.labels = self.read_protein_data(file_path)

    def __len__(self):
        return len(self.protains)

    def __getitem__(self, item):
        protein = self.protains[item]

        label = self.labels[item]

        return protein['sequence'], label,protein["identifier"]
    def read_protein_data(self,file_path):
        proteins = []
        labels = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

            lens = len(lines)
            for i in range(0, lens, 3):
                # 处理标识行
                identifier = lines[i].strip()[1:]

                # 处理蛋白质序列行
                sequence = lines[i + 1].strip()

                label_str = lines[i + 2].strip()

                #如果序列长度大于1200，就将其分为两条，减少显存的使用。
                if len(sequence) > 1200 and len(sequence) != 1203 and 'Test' not in file_path:

                    identifier1 = identifier

                    sequence1 = sequence[0:1200]
                    sequence = sequence[1200:]

                    label_str1 = label_str[0:1200]
                    label_str = label_str[1200:]

                    label_list1 = [int(label_str1[index]) for index in range(0, len(label_str1))]
                    label_list = [int(label_str[index]) for index in range(0, len(label_str))]

                    proteins.append({'identifier': identifier1+'_first', 'sequence': sequence1})

                    labels.append(label_list1)

                    proteins.append({'identifier': identifier+'_second', 'sequence': sequence})

                    labels.append(label_list)
                    continue

                label_list = [int(label_str[index]) for index in range(0,len(label_str))]

                proteins.append({'identifier': identifier, 'sequence': sequence})

                labels.append(label_list)

            return proteins, labels











