import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond

def collate_autoreg(batch):
    notnone_batches = [b for b in batch if b is not None]
    
    motionbatch_0 = [b['motion_0'] for b in notnone_batches]
    if 'lengths_0' in notnone_batches[0]:
        lenbatch_0 = [b['lengths_0'] for b in notnone_batches]
    else:
        lenbatch_0 = [len(b['motion_0'][0][0]) for b in notnone_batches]
    motionbatchTensor_0 = collate_tensors(motionbatch_0)
    lenbatchTensor_0 = torch.as_tensor(lenbatch_0)
    maskbatchTensor_0 = lengths_to_mask(lenbatchTensor_0, motionbatchTensor_0.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
    textbatch_0 = [b['text_0'] for b in notnone_batches]
    tokenbatch_0 = [b['tokens_0'] for b in notnone_batches]
    motion_0 = motionbatchTensor_0
    cond_0 = {'y': {'motion_prev':None, 'mask_prev': None, 'lengths_prev': None, 'text_prev': None, 'tokens_prev': None,
                                        'mask_curr': maskbatchTensor_0, 'lengths_curr': lenbatchTensor_0, 'text_curr': textbatch_0, 'tokens_curr': tokenbatch_0 }}
    
    motionbatch_1 = [b['motion_1'] for b in notnone_batches]
    if 'lengths_1' in notnone_batches[0]:
        lenbatch_1 = [b['lengths_1'] for b in notnone_batches]
    else:
        lenbatch_1 = [len(b['motion_1'][0][0]) for b in notnone_batches]
    motionbatchTensor_1 = collate_tensors(motionbatch_1)
    lenbatchTensor_1 = torch.as_tensor(lenbatch_1)
    maskbatchTensor_1 = lengths_to_mask(lenbatchTensor_1, motionbatchTensor_1.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
    textbatch_1 = [b['text_1'] for b in notnone_batches]
    tokenbatch_1 = [b['tokens_1'] for b in notnone_batches]
    motion_1 = motionbatchTensor_1
    cond_1 = {'y': {'motion_prev':motion_0, 'mask_prev': maskbatchTensor_0, 'lengths_prev': lenbatchTensor_0, 'text_prev': textbatch_0, 'tokens_prev': tokenbatch_0,
                                            'mask_curr': maskbatchTensor_1, 'lengths_curr': lenbatchTensor_1, 'text_curr': textbatch_1, 'tokens_curr': tokenbatch_1 }}
    return motion_0, cond_0, motion_1, cond_1

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)

def t2m_autoreg_collate(batch):
    adapted_batch = [{
        'text_0': b[0],
        'motion_0': torch.tensor(b[1].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'lengths_0': b[2],
        'tokens_0': b[3],
        'text_1': b[4],
        'motion_1': torch.tensor(b[5].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'lengths_1': b[6],
        'tokens_1': b[7],
    } for b in batch]
    return collate_autoreg(adapted_batch)


