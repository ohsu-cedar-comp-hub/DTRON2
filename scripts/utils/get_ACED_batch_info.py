import os 

#need to get all folder names, 1 layer down within BatchN/SubractedRegisteredImages
ACED_path = "/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/ACED"
batch_paths = [os.path.join(os.path.join(ACED_path,x),'SubtractedRegisteredImages') for x in os.listdir(ACED_path) if os.path.isdir(os.path.join(ACED_path,x)) and "Batch" in x]

batch_groups = [list(set([x[:x.find("_scene")] for x in os.listdir(y) if os.path.isdir(os.path.join(y,x))])) for y in batch_paths]

batch_dic = {x:i for (i,y) in enumerate(batch_groups) for x in y}

print(batch_dic)