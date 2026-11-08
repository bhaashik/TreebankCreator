"""Offline regression tests for dangerous systematic Hindi-to-target alignment errors."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ud_treebank_lab.corpus_config import read_corpus_config
from ud_treebank_lab.corpus_morph import parse_apertium_candidates, read_unimorph
from ud_treebank_lab.corpus_refinement import run_corpus_refinement
from ud_treebank_lab.utils import jsonl
from ud_treebank_lab.bilingual_config import read_config, ensure_corpus_preprocessing
from ud_treebank_lab.bilingual_data import soft_alignment


def _data(tmp_path: Path, *, two: bool = True):
    # Intentionally wrong alignment of Hindi ने (index 4) to Bhojpuri name (3).
    source = ("# sent_id = 1\n# text = इसका निर्माण नवाब शाहजहाँ ने करवाया था ।\n"
              "1\tइसका\tयह\tPRON\t_\tPronType=Prs\t2\tnmod\t_\t_\n"
              "2\tनिर्माण\tनिर्माण\tNOUN\t_\t_\t6\tobj\t_\t_\n"
              "3\tनवाब\tनवाब\tNOUN\t_\t_\t4\tcompound\t_\t_\n"
              "4\tशाहजहाँ\tशाहजहाँ\tPROPN\t_\t_\t6\tnsubj\t_\t_\n"
              "5\tने\tने\tADP\t_\t_\t4\tcase\t_\t_\n"
              "6\tकरवाया\tकरवाना\tVERB\t_\t_\t0\troot\t_\t_\n"
              "7\tथा\tहोना\tAUX\t_\t_\t6\taux\t_\t_\n"
              "8\t।\t।\tPUNCT\t_\t_\t6\tpunct\t_\t_\n\n")
    if two:
        source += source.replace('sent_id = 1','sent_id = 2')
    (tmp_path/'source.conllu').write_text(source,encoding='utf8')
    rows = [{"id":"1","tokens":["एकर","निर्माण","नवाब","शाहजेहन","कइले","रहलें","।"]}]
    if two: rows += [{**rows[0],"id":"2"}]
    (tmp_path/'target.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    wrong = {"id":"1","pairs":[[0,0],[1,1],[2,2],[4,3],[5,4],[6,5],[7,6]]}
    raw = [wrong]
    if two: raw += [{**wrong,"id":"2"}]
    (tmp_path/'initial.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in raw),encoding='utf8')
    return rows


def _config(tmp_path, *, align="supplied", morph="skip"):
    config = {"source_ud":"source.conllu", "targets":"target.jsonl", "output_dir":"corpus",
              "resources":{"alignment":{"mode":align, **({"path":"initial.jsonl"} if align=="supplied" else {})},
                           "apertium":{"mode": morph, **({"path":"morph.jsonl"} if morph=="supplied" else {})},
                           "unimorph":{"mode":"skip"}},
              "refinement":{"iterations":2,"anchor_similarity":.68,"max_alternatives":3}}
    file = tmp_path/'corpus.yaml'
    file.write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf8')
    return file


def test_bhojpuri_name_corrected_without_running_models(tmp_path,monkeypatch):
    _data(tmp_path)
    cfg = _config(tmp_path)
    from ud_treebank_lab import corpus_refinement as cr
    monkeypatch.setattr(cr,"run_alignment", lambda *a,**kw: (_ for _ in ()).throw(AssertionError("shouldn't align")))
    monkeypatch.setattr(cr,"generate_apertium_lattice",lambda *a,**kw: (_ for _ in ()).throw(AssertionError("shouldn't analyze")))
    original = (tmp_path/'initial.jsonl').read_bytes()
    result = run_corpus_refinement(cfg)
    records = list(jsonl(result['alignments']))
    for rec in records:
        # Hindi Shah Jahan source token 3 (zero-indexed) should override erroneous ने token 4.
        assert [3,3] in rec['pairs'], rec
        assert [4,3] not in rec['pairs'], rec
        assert 4 in rec['null_source'], rec
        a, rel = soft_alignment(rec,8,7)
        assert a[3,3] > a[3,4]
        assert 0 < rel[3] <= 1
    assert (tmp_path/'initial.jsonl').read_bytes() == original
    assert run_corpus_refinement(cfg)['status']=='already-complete'
    audit = list(jsonl(tmp_path/'corpus/corrections.jsonl'))
    assert audit[0]['changed'] and [4,3] in audit[0]['initial_pairs']


def test_no_external_resource_defaults_and_strict_paths(tmp_path):
    _data(tmp_path)
    cfg = {"source_ud":"source.conllu", "targets":"target.jsonl", "output_dir":"out"}
    file = tmp_path/'config.yaml'
    file.write_text(yaml.safe_dump(cfg))
    assert read_corpus_config(file)['resources']['alignment']['mode']=='skip'
    assert read_corpus_config(file)['resources']['apertium']['mode']=='skip'
    result = run_corpus_refinement(file)
    assert result['status']=='complete'
    assert len(list(jsonl(result['alignments'])))==2
    cfg['resources']={'alignment':{'mode':'supplied','path':'missing.jsonl'}}
    file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(FileNotFoundError):read_corpus_config(file)
    cfg['resources']={'alignment':{'mode':'generate','path':'initial.jsonl'}}
    file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):read_corpus_config(file)


def test_supplied_ambiguous_morphology_and_unimorph(tmp_path):
    _data(tmp_path)
    morph = [{"id":str(i),"analyses":[[],[],[],[
        {"lemma":"शाहजेहन","upos":"PROPN","feats":{}},
        {"lemma":"शाहजेहन","upos":"NOUN","feats":{}}],[],[],[]]} for i in (1,2)]
    (tmp_path/'morph.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in morph),encoding='utf8')
    (tmp_path/'um.tsv').write_text('शाहजेहन\tशाहजेहन\tPROPN;SG\n',encoding='utf8')
    cfg=_config(tmp_path,morph='supplied')
    doc=yaml.safe_load(cfg.read_text());doc['resources']['unimorph']={'path':'um.tsv','mode':'supplied'}
    cfg.write_text(yaml.safe_dump(doc,allow_unicode=True))
    run_corpus_refinement(cfg)
    rows=list(jsonl(tmp_path/'corpus/morph_candidates.jsonl'))
    assert len(rows[0]['words'][3]['candidates']) == 3
    assert rows[0]['words'][3]['status']=='resource_ambiguous'
    assert 'शाहजेहन' in read_unimorph(tmp_path/'um.tsv')
    assert len(parse_apertium_candidates('^शाहजेहन/शाहजेहन<np><sg>/शाहजेहन<n><sg>$')) == 2


def test_changing_initial_alignment_invalidates_outputs_without_modifying_inputs(tmp_path):
    _data(tmp_path)
    cfg = _config(tmp_path)
    first=run_corpus_refinement(cfg)
    original=first['signature']
    path=tmp_path/'initial.jsonl'
    text=path.read_text().replace('[[0, 0]', '[[0, 0]')
    path.write_text(text+'\n')  # fingerprint change, legitimate JSONL blank line
    second=run_corpus_refinement(cfg)
    assert second['status']=='complete' and second['signature'] != original


def test_invalid_indices_rejected(tmp_path):
    _data(tmp_path,two=False)
    file=tmp_path/'initial.jsonl'
    file.write_text(json.dumps({'id':'1','pairs':[[20,3]]})+'\n')
    cfg=_config(tmp_path)
    with pytest.raises(ValueError,match='Invalid alignment index'):
        run_corpus_refinement(cfg)


def test_bilingual_config_runs_corpus_only_when_explicitly_selected(tmp_path):
    _data(tmp_path)
    cfg=_config(tmp_path)
    bilingual={"language":"bho", "source_ud":"source.conllu", "targets":"target.jsonl",
               "alignments":"corpus/alignments.refined.jsonl", "corpus_preprocess":"corpus.yaml",
               "source_cache":"s", "target_cache":"t", "output_dir":"runs"}
    file=tmp_path/'bilingual.yaml';file.write_text(yaml.safe_dump(bilingual))
    resolved=read_config(file)
    assert not (tmp_path/'corpus/alignments.refined.jsonl').exists()
    result=ensure_corpus_preprocessing(resolved)
    assert result['status']=='complete'
    assert ensure_corpus_preprocessing(resolved)['status']=='already-complete'
    bilingual['alignments']='initial.jsonl'
    file.write_text(yaml.safe_dump(bilingual))
    with pytest.raises(ValueError, match='must point at corpus'):
        ensure_corpus_preprocessing(read_config(file))


def test_repeated_cross_script_name_can_override_bad_alignment(tmp_path):
    # No lexical/character resemblance between the Hindi proper name and
    # its Latin-script rendering. The repeated sentence-level association,
    # plus the behavior of common Hindi ने across sentences, must matter.
    source_rows=[]; targets=[]; aligns=[]
    for i in range(1, 9):
        name='गंगाधर' if i <= 4 else f'अन्य{i}'
        target_name='Gangadhar' if i <= 4 else f'Other{i}'
        source_rows.append(
            f'# sent_id = {i}\n'
            f'1\t{name}\t{name}\tPROPN\t_\t_\t3\tnsubj\t_\t_\n'
            f'2\tने\tने\tADP\t_\t_\t1\tcase\t_\t_\n'
            f'3\tबनाया\tबनाना\tVERB\t_\t_\t0\troot\t_\t_\n\n')
        targets.append({'id':str(i),'tokens':[target_name,'produced']})
        aligns.append({'id':str(i),'pairs':[[1,0] if i==1 else [0,0],[2,1]]})
    (tmp_path/'source.conllu').write_text(''.join(source_rows),encoding='utf8')
    (tmp_path/'target.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in targets))
    (tmp_path/'initial.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in aligns))
    cfg=_config(tmp_path)
    result=run_corpus_refinement(cfg)
    first=next(jsonl(result['alignments']))
    assert [0,0] in first['pairs'],first
    assert [1,0] not in first['pairs'],first
    correction=next(jsonl(tmp_path/'corpus/corrections.jsonl'))
    assert correction['evidence'][0]['alternatives'][0]['sentence_cooccurrence']>0


def test_generators_are_called_only_with_explicit_generate(tmp_path, monkeypatch):
    _data(tmp_path)
    config=_config(tmp_path,align='generate',morph='generate')
    doc=yaml.safe_load(config.read_text())
    doc['resources']['apertium']['analyzer_bin']='initial.jsonl'  # mock analyzer file
    doc['resources']['apertium']['binary']='python'  # present test-only executable
    doc['resources']['alignment']['backend']='identity'  # no SimAlign setup needed
    config.write_text(yaml.safe_dump(doc))
    from ud_treebank_lab import corpus_resources as cr
    calls=[]
    def fake_alignment(src,tgt,dst,**kwargs):
        calls.append('alignment')
        Path(dst).write_bytes((tmp_path/'initial.jsonl').read_bytes())
    def fake_morph(tgt,dst,analyzer_bin,**kwargs):
        calls.append('morph')
        Path(dst).write_text(''.join(json.dumps({'id':str(i),'analyses':[[]]*7})+'\n' for i in (1,2)))
    monkeypatch.setattr(cr,'run_alignment',fake_alignment)
    monkeypatch.setattr(cr,'generate_apertium_lattice',fake_morph)
    result=run_corpus_refinement(config)
    assert calls==['alignment','morph']
    assert run_corpus_refinement(config)['status']=='already-complete'
    assert calls==['alignment','morph']
    assert Path(result['alignments']).is_file()


def test_manifest_infers_refined_alignment_path(tmp_path):
    from ud_treebank_lab.bilingual_manifest import make_configs
    _data(tmp_path)
    corpus=_config(tmp_path)
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps({'language':'bho','targets':'target.jsonl',
                                   'corpus_preprocess':'corpus.yaml'})+'\n')
    result=make_configs(str(manifest),str(tmp_path/'source.conllu'),
                        str(tmp_path/'configs'),str(tmp_path/'cache'),str(tmp_path/'runs'))
    cfg=read_config(result['configs'][0])
    assert cfg['corpus_preprocess']==str(corpus)
    assert cfg['alignments']==str(tmp_path/'corpus/alignments.refined.jsonl')
    assert not Path(cfg['alignments']).exists()  # manifest creation doesn't run stage


def test_full_bilingual_training_uses_refined_alignment_offline(tmp_path):
    import torch
    from ud_treebank_lab.bilingual_training import _pairs, _source_rows, train_bilingual
    from ud_treebank_lab.bilingual_cache import cache_signature
    from ud_treebank_lab.utils import write_json
    _data(tmp_path)
    _config(tmp_path)
    cfg={'language':'bho','source_ud':'source.conllu','targets':'target.jsonl',
         'corpus_preprocess':'corpus.yaml',
         'alignments':'corpus/alignments.refined.jsonl',
         'source_cache':'s_cache','target_cache':'t_cache','output_dir':'trained',
         'model':'no-internet-fake','epochs':1,'batch_size':1,'accumulate':1,
         'save_steps':1,'hidden':16,'layers':1,'shard_size':2,'max_subtokens':64,
         'dropout':0.,'alignment_dropout':0.,'device':'cpu','max_words':70}
    f=tmp_path/'model.yaml';f.write_text(yaml.safe_dump(cfg))
    resolved=read_config(f)
    ensure_corpus_preprocessing(resolved)
    rows_source=_source_rows(resolved)
    rows_target=[(p.sid,p.tokens) for p in _pairs(resolved)]
    for folder, rows in [(tmp_path/'s_cache', rows_source),(tmp_path/'t_cache', rows_target)]:
        folder.mkdir()
        write_json(folder/'manifest.json',{'signature':cache_signature(rows,'no-internet-fake',64,2),
            'model':'no-internet-fake','max_subtokens':64,'shard_size':2,'num_sentences':2,
            'pooling':'first-subword-v1'})
        torch.save({sid:torch.randn(len(tokens),12).half() for sid,tokens in rows},
                   folder/'part_00000.pt')
    result=train_bilingual(str(f),auto_cache=False,device='cpu')
    assert result['status']=='complete'
    assert (tmp_path/'trained/best.pt').exists()
    assert train_bilingual(str(f),auto_cache=False,device='cpu')['status']=='already-complete'


def test_function_word_suffix_requires_contrasting_host_forms(tmp_path):
    # Artificial corpus: a stable host lemma has two target suffix variants,
    # and Hindi ने occurs only with one. The stage should propose (not assert)
    # a possible bound realization and refrain from inventing target word arcs.
    src=[]; tgt=[]; aln=[]
    for i in range(1, 7):
        with_particle=(i<=3)
        src.append(f'# sent_id = {i}\n1\tघनश्याम\tघनश्याम\tPROPN\t_\t_\t0\troot\t_\t_\n'+
                   ('2\tने\tने\tADP\t_\t_\t1\tcase\t_\t_\n' if with_particle else '')+'\n')
        tgt.append({'id':str(i),'tokens':['घरवाले' if with_particle else 'घरवाला']})
        aln.append({'id':str(i),'pairs':[[0,0]]})
    (tmp_path/'source.conllu').write_text(''.join(src),encoding='utf8')
    (tmp_path/'target.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in tgt))
    (tmp_path/'initial.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in aln))
    cfg=_config(tmp_path)
    run_corpus_refinement(cfg)
    patterns=json.loads((tmp_path/'corpus/function_realizations.json').read_text())
    entry=patterns['ने|ADP|case']
    assert entry['null_or_bound']==3
    # A candidate suffix is only proposed after the required positive/negative
    # contrast, not because the source particle merely co-occurred with it.
    assert entry['candidate_affixes'], entry
    assert entry['candidate_affixes'][0]['association_only'] is True
    bound=list(jsonl(tmp_path/'corpus/bound_morpheme_hypotheses.jsonl'))
    assert bound[0]['candidates'], bound[0]
    assert bound[0]['candidates'][0]['source_form']=='ने'
