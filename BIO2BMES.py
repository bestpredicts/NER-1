def bio2bmes():
    """将bio标注转为bmes标注
    """
    # read bio file
    data_path = ROOT_PATH / 'data_src'
    with open(data_path / f'{}.bio', 'r', encoding='utf-8') as sen_f, \
            open(data_path / f'{}_tags.bio', 'r', encoding='utf-8') as tag_f:
        sens = [''.join(line.strip().split(' ')) for line in sen_f]
        tags = [line.strip().split(' ') for line in tag_f]
        
        # sanity check
        assert len(sens) == len(tags)
        for s, t in zip(sens, tags):
            assert len(s) == len(t)

    # write to bmes
    with open(data_path / f'{}.bmes', 'w', encoding='utf-8') as f:
        for sen, tag in zip(sens, tags):
            for idx, (w, t) in enumerate(zip(sen, tag)):
                if t[0] == 'I':
                    if idx == len(sen) - 1 or tag[idx + 1][0] in ('O', 'B'):
                        f.write(w + '\t' + 'E' + t[1:] + '\n')
                    else:
                        f.write(w + '\t' + 'M' + t[1:] + '\n')
                elif t[0] == 'B' and len(tag) == 1:
                    f.write(w + '\t' + 'S' + t[1:] + '\n')
                else:
                    f.write(w + '\t' + t + '\n')
            f.write('\n')