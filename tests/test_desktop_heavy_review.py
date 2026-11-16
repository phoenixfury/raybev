"""Desktop reviewer data integrity, persistence, projection and export tests."""
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import pytest

spec=importlib.util.spec_from_file_location('desktop_review',Path(__file__).resolve().parents[1]/'scripts/review_heavy_annotations.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def test_named_ui_fonts_supports_older_tkinter_signature():
    root=object()
    fonts={'TkDefaultFont':object(),'TkHeadingFont':object()}
    class OldTkFont:
        @staticmethod
        def names(actual_root):
            assert actual_root is root
            return tuple(fonts)

        @staticmethod
        def nametofont(name):
            return fonts[name]

    assert review.named_ui_fonts(OldTkFont,root)==fonts


def test_explicit_fonts_override_theme_defaults_and_preserve_headings():
    class Widget:
        def __init__(self,kind,options,children=()):
            self.kind=kind;self.options=options;self.children=children
        def keys(self):return self.options.keys()
        def cget(self,key):return self.options[key]
        def winfo_class(self):return self.kind
        def winfo_children(self):return self.children
        def configure(self,**settings):self.options.update(settings)
    label=Widget('TLabel',{'font':'TkSmallCaptionFont','style':''})
    entry=Widget('TEntry',{'font':'TkTextFont','style':''})
    combo=Widget('TCombobox',{'font':'TkTextFont','style':''})
    button=Widget('TButton',{'style':''})
    heading=Widget('TLabel',{'font':'Heading','style':''})
    root=Widget('Tk',{},[label,entry,combo,button,heading])
    review.apply_text_fonts(root,'ReviewFont','Heading')
    for widget in (label,entry,combo):
        assert widget.options['font']=='ReviewFont'
        assert widget.options['style']=='Review.'+widget.kind
    assert button.options['style']=='Review.TButton'
    assert heading.options['font']=='Heading'


def test_scalable_font_selection_rejects_bitmap_fallback():
    from types import SimpleNamespace
    class Font:
        def __init__(self,root,family,size):self.family=family;self.size=size
        def configure(self,size):self.size=size
        def metrics(self,key):return 13 if self.family=='fixed' else abs(self.size)+3
        def actual(self,key):return self.family
    fake=SimpleNamespace(Font=Font,families=lambda root:['fixed','Liberation Sans'])
    assert review.scalable_font_family(fake,object())=='Liberation Sans'
    fake.families=lambda root:['fixed']
    with pytest.raises(RuntimeError,match='no scalable font'):
        review.scalable_font_family(fake,object())


def test_font_changes_update_pixel_sizes_styles_and_visible_readout():
    from types import SimpleNamespace
    class Font:
        def configure(self,**settings):self.settings=settings
    style_calls=[];titles=[];readouts=[]
    root=SimpleNamespace(keys=lambda:[],winfo_class=lambda:'Tk',winfo_children=lambda:[],title=titles.append)
    window=SimpleNamespace(root=root,font_family='Liberation Sans',fonts={'TkDefaultFont':Font()},ui_font=Font(),camera_font=Font(),
        style=SimpleNamespace(configure=lambda name,**settings:style_calls.append((name,settings))),
        font_readout=SimpleNamespace(set=readouts.append))
    review.ReviewWindow.set_font_size(window,20)
    assert window.ui_font.settings['size']==-27
    review.ReviewWindow.set_font_size(window,32)
    assert window.ui_font.settings['size']==-43
    assert window.camera_font.settings['size']==-45
    assert window.fonts['TkDefaultFont'].settings['size']==-43
    assert readouts[-1]=='32 pt'
    assert review.REVIEW_UI_VERSION in titles[-1]
    assert any(name=='Review.TButton' for name,settings in style_calls)


def test_large_font_control_layout_wraps_without_losing_widgets():
    assert review.flow_layout([110,110,110],250)==[(0,0),(0,1),(1,0)]
    assert review.flow_layout([300,100],250)==[(0,0),(1,0)]
    assert review.flow_layout([],250)==[]
    assert review.overlay_font(24) is not None


def test_preview_enlarges_small_crop_and_preserves_aspect_ratio():
    image=Image.new('RGB',(100,50))
    assert review.fit_preview(image,800,500).size==(800,400)
    assert review.fit_preview(image,40,100).size==(40,20)
    assert review.fit_preview(image,0,0).size==(1,1)


def test_wrapped_controls_are_raised_above_later_created_row_frames():
    from types import SimpleNamespace
    calls=[]
    class Widget:
        def pack_forget(self):pass
        def winfo_reqwidth(self):return 100
        def winfo_reqheight(self):return 30
        def pack(self,**kwargs):calls.append('pack')
        def lift(self):calls.append('lift')
    children=[Widget(),Widget()]
    class Frame(Widget):
        def __init__(self,*args):pass
        def winfo_children(self):return children
        def winfo_width(self):return 400
        def bind(self,*args):pass
    root=SimpleNamespace(after_idle=lambda callback:callback())
    window=SimpleNamespace(root=root,ttk=SimpleNamespace(Label=type('Label',(),{}),Frame=Frame),flow_refresh=[])
    review.ReviewWindow.wrap_controls(window,Frame())
    assert calls[-4:]==['pack','lift','pack','lift']


@pytest.fixture
def session(tmp_path):
    image=tmp_path/'images/shared.png';image.parent.mkdir()
    Image.new('RGB',(800,640),(85,110,140)).save(image)
    def row(token,split,label,uid,x):
        return dict(token=token,metainfo=dict(temporal_split=split,sequence_id='bag',kept='yes'),
            cam_info={'cam_fisheye':{c:{'cam_path':'images/shared.png'} for c in review.CAMERAS}},
            annos=dict(gt_boxes=np.array([[x,2.,0.,8.,2.5,3.,.2],[7.,-4.,-.5,4.,1.8,1.5,.4]],np.float32),
                gt_names=np.array([label,'Car']),gt_ids=np.array([uid,100+uid]),
                gt_mesh=np.array(['detzero.'+label.lower(),'detzero.car'])))
    paths={}
    for split,rows in [('train',[row('t1','train','Truck',7,10.),row('t2','train','Van',8,15.)]),
                       ('val',[row('v1','val','Truck',7,10.)])]:
        paths[split]=tmp_path/f'{split}.pkl'
        paths[split].write_bytes(pickle.dumps(dict(metainfo={'unchanged':True},data_list=rows),protocol=4))
    ds=review.RosaCombinedDataset(tmp_path,paths['train'],cfg=review.SparseConfig(architecture_version=5),
                                  calibration=review.DEFAULT_CALIBRATION)
    records={str(r['token']):[dict(gt=b[0].tolist(),bbox=[20.,30.,100.,150.],camera=0,
        verified=True,source='manual_2d',edge_valid=[True]*4,occluded=False,truncated=False)]
        for r,b in zip(ds.rows,ds.boxes)}
    from raybev.rosa_dataset import training_boxes_sha256
    ext=tmp_path/'extents.json'
    ext.write_text(json.dumps(dict(type='raybev_verified_extents_v1',split='train',
        training_boxes_sha256=training_boxes_sha256(ds),camera_calibration_sha256=review.sha(review.DEFAULT_CALIBRATION),
        coordinates='network_pixels',image_size=[800,640],camera_order=list(review.CAMERAS),
        frames=records,counts={'truck':1,'van':1},evidence='preserve me')))
    return review.ReviewSession(tmp_path,paths['train'],paths['val'],ext,review.DEFAULT_CALIBRATION,tmp_path/'state.json')


def test_desktop_layout_with_display(session,monkeypatch):
    tk=pytest.importorskip('tkinter')
    try:
        probe=tk.Tk()
    except tk.TclError:
        pytest.skip('A working desktop display is required for the Tk layout smoke test')
    probe.destroy()
    window=review.ReviewWindow(session,session.root/'release',font_size=20)
    try:
        window.root.update()
        assert window.ui_font.cget('size')==-27
        assert window.ui_font.metrics('linespace')>=25
        assert window.track.get() and window.both.get()
        assert window.controls_canvas.winfo_viewable()
        assert all(canvas.winfo_height()>100 for canvas in window.canvases)
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        buttons=[w for w in descendants(window.root) if w.winfo_class()=='TButton']
        assert buttons and all(button.winfo_ismapped() for button in buttons)
        assert next(w for w in buttons if w.cget('text')=='1: Car').winfo_viewable()
        height=buttons[0].winfo_reqheight()
        linespace=window.ui_font.metrics('linespace')
        window.set_font_size(32)
        window.root.update()
        assert window.ui_font.cget('size')==-43
        assert window.ui_font.metrics('linespace')>linespace
        assert buttons[0].winfo_reqheight()>height
        monkeypatch.setattr(window.messagebox,'askyesno',lambda *args,**kwargs:True)
        window.commit('car')
        assert session.current(session.by_id['train:t1:0'])['label']=='car'
        assert session.current(session.by_id['val:v1:0'])['label']=='car'
        assert window.ids[window.position]=='train:t2:0'
        window.position=window.ids.index('train:t1:0');window.show()
        removal=next(w for w in buttons if w.cget('text')=='Remove from this frame')
        removal.invoke()
        assert session.current(session.by_id['train:t1:0'])['status']=='removed'
        assert session.current(session.by_id['val:v1:0'])['status']=='reviewed'
        assert 'train:t1:0' not in window.ids
    finally:
        window.root.destroy()


def reopen(s):
    return review.ReviewSession(s.root,s.paths['train'],s.paths['val'],s.extents,s.calibration,s.state_path)


def test_queue_progress_track_scope_and_persisted_undo(session):
    s=session;assert s.summary()['queued']==3
    key='train:t1:0'
    assert s.apply(key,'car',track=True)==1
    assert s.current(s.by_id['val:v1:0'])['label']=='truck'
    assert s.apply(key,'van',track=True,both_splits=True)==2
    assert s.current(s.by_id['val:v1:0'])['label']=='van'
    restored=reopen(s);assert restored.undo()
    assert restored.current(restored.by_id[key])['label']=='car'
    assert restored.current(restored.by_id['val:v1:0'])['label']=='truck'
    assert restored.undo() and restored.summary()['reviewed']==0
    assert not restored.undo()


def test_entire_track_includes_car_frames_without_crossing_bags_or_unknown_ids(session):
    import copy
    s=session;doc=copy.deepcopy(s.documents['train'])
    for token,sequence,track,label in [('t3','bag',7,'Car'),('t4','other',7,'Car'),
                                      ('t5','bag',-1,'Truck'),('t6','bag',-1,'Truck')]:
        row=copy.deepcopy(doc['data_list'][0]);row['token']=token
        row['metainfo']['sequence_id']=sequence
        row['annos']['gt_ids'][0]=track;row['annos']['gt_names'][0]=label
        doc['data_list'].append(row)
    s.paths['train'].write_bytes(pickle.dumps(doc,protocol=4))
    s=reopen(s)
    assert 'train:t3:0' in s.by_id and 'train:t4:0' not in s.by_id
    assert s.apply('train:t1:0','bus',track=True,both_splits=True)==3
    restored=reopen(s)
    for key in ('train:t1:0','train:t3:0','val:v1:0'):
        assert restored.current(restored.by_id[key])['label']=='bus'
    assert restored.targets('train:t5:0',track=True,both_splits=True)==['train:t5:0']
    assert restored.undo()
    assert restored.current(restored.by_id['train:t3:0'])['label']=='car'


def issue_fixture(session):
    s = review.ReviewSession(session.root,session.paths['train'],session.paths['val'],session.extents,
                             session.calibration,session.root/'followup_state.json',classes=review.CLASSES)
    entries = [dict(split='train',token=token,index=index,sequence='bag',track_id=track,
                    reason='Check geometry/association',status='unresolved')
               for token,index,track in [('t1',1,'107'),('t2',0,'8')]]
    ledger = s.root/'issues.jsonl'
    ledger.write_text(''.join(json.dumps(entry)+'\n' for entry in entries))
    contract = s.root/'contract.json'
    contract.write_text(json.dumps(dict(files={
        'fisheye3dod_infos_train.pkl':s.identity['annotations']['train'],
        'fisheye3dod_infos_val.pkl':s.identity['annotations']['val'],
        'train_yolo_extents_v1.json':s.identity['extents']},
        calibration_sha256=s.identity['calibration'])))
    return s,ledger,contract


def test_remaining_issues_desktop_scope_reason_and_frame_only_default(session,monkeypatch):
    tk=pytest.importorskip('tkinter')
    try:probe=tk.Tk()
    except tk.TclError:pytest.skip('A working desktop display is required')
    probe.destroy()
    s,ledger,contract=issue_fixture(session);s.remaining_issues(ledger,contract)
    window=review.ReviewWindow(s,s.root/'v6',font_size=20)
    try:
        window.root.update()
        assert not window.track.get() and window.status.get()=='unknown'
        assert set(window.ids)=={'train:t1:1','train:t2:0'}
        assert window.issue.winfo_viewable() and 'Check geometry/association' in window.issue.cget('text')
        assert 'Total 2' in window.progress.cget('text')
        assert all(canvas.winfo_height()>100 for canvas in window.canvases)
        window.commit('van')
        assert window.ids==['train:t1:1']
        assert s.current(s.by_id['val:v1:0'])['status']=='unreviewed'
        monkeypatch.setattr(window.messagebox,'askyesno',lambda *args,**kwargs:True)
        window.remove_frame()
        assert not window.ids and s.summary()['removed']==1
        window.status.set('removed');window.filter()
        assert window.ids==['train:t1:1']
        window.undo()
        window.status.set('all');window.filter()
        assert set(window.ids)=={'train:t1:1','train:t2:0'}
        assert s.summary()['removed']==0
    finally:window.root.destroy()


def test_remaining_issues_select_final_indices_and_export_complete_dataset(session):
    s,ledger,contract = issue_fixture(session)
    s.remaining_issues(ledger,contract)
    assert {r['id'] for r in s.queue_records()} == {'train:t1:1','train:t2:0'}
    assert s.queue_records()[0]['original_class']=='van'
    assert s.summary()['queued']==2 and s.summary()['unknown_remaining']==2
    assert len(s.records)==6  # Full pool is retained for optional track edits and export.
    s.apply('train:t1:1','car',dimensions=[4.,1.8,1.5])
    s.remove('train:t2:0')
    assert s.summary()['unknown_remaining']==0 and s.summary()['removed']==1
    restored = review.ReviewSession(s.root,s.paths['train'],s.paths['val'],s.extents,s.calibration,
                                   s.state_path,classes=review.CLASSES)
    restored.remaining_issues(ledger,contract)
    assert restored.summary()==s.summary()
    report = restored.export(s.root/'followup_release')
    assert report['splits']['train']['all_boxes_checked']==3
    assert report['splits']['val']['all_boxes_checked']==2
    assert review.verify_export(s.root,s.root/'followup_release',s.calibration)['all_boxes_checked']==5


@pytest.mark.parametrize('fault',['input_hash','track_id','index','duplicate','empty'])
def test_remaining_issues_reject_stale_or_ambiguous_ledger(session,fault):
    s,ledger,contract = issue_fixture(session)
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    if fault=='input_hash':
        doc=json.loads(contract.read_text());doc['files']['fisheye3dod_infos_train.pkl']='wrong'
        contract.write_text(json.dumps(doc))
    elif fault=='track_id':entries[0]['track_id']='wrong'
    elif fault=='index':entries[0]['index']=99
    elif fault=='duplicate':entries.append(entries[0])
    else:entries=[]
    ledger.write_text(''.join(json.dumps(entry)+'\n' for entry in entries))
    with pytest.raises(ValueError):s.remaining_issues(ledger,contract)
    assert s.issue_records is None and not s.state_path.exists()


def test_prior_unresolved_ledger_is_prioritized_until_track_is_reviewed(session):
    s=session
    ledger=s.root/'annotation_review.jsonl'
    ledger.write_text(json.dumps(dict(split='train',token='t2',index=0,sequence='bag',track_id='8',
                                     status='unresolved',reason='Occluded in original review'))+'\n')
    s=reopen(s)
    unknown=s.by_id['train:t2:0']
    assert s.summary()['unknown_remaining']==1
    assert s.unknown(unknown) and s.review_priority(unknown)==0
    assert sorted(s.records,key=s.review_priority)[0]['id']==unknown['id']
    s.apply(unknown['id'],'van',track=True,both_splits=True)
    assert not s.unknown(unknown) and s.summary()['unknown_remaining']==0
    s.apply('train:t1:0',uncertain=True)
    assert s.unknown(s.by_id['train:t1:0'])
    assert s.undo() and s.undo()
    assert s.unknown(unknown)


def test_track_export_synchronizes_both_splits_and_yolo(session):
    s=session
    assert s.apply('train:t1:0','bus',track=True,both_splits=True)==2
    out=s.root/'track_release';s.export(out)
    for split in ('train','val'):
        doc=review.load_annotations(out/f'fisheye3dod_infos_{split}.pkl')
        assert doc['data_list'][0]['annos']['gt_names'][0]=='Bus'
        np.testing.assert_array_equal(doc['data_list'][0]['annos']['gt_ids'],
                                      s.documents[split]['data_list'][0]['annos']['gt_ids'])
    ext=json.loads((out/'train_yolo_extents_v1.json').read_text())
    assert ext['frames']['t1'][0]['gt'][7]==5.
    assert review.verify_export(s.root,out,s.calibration)['status']=='passed'


def test_single_frame_removal_persists_and_track_edits_do_not_restore_it(session):
    s=session;s.apply('train:t1:0','van',track=True,both_splits=True)
    s.remove('train:t1:0',note='Ghost box in this frame')
    s=reopen(s)
    assert s.summary()['removed']==1
    assert s.current(s.by_id['val:v1:0'])['status']=='reviewed'
    assert s.targets('train:t1:0',track=True,both_splits=True)==['val:v1:0']
    assert s.apply('train:t1:0','bus',track=True,both_splits=True)==1
    assert s.current(s.by_id['train:t1:0'])['status']=='removed'
    assert s.undo() and s.undo()
    assert s.current(s.by_id['train:t1:0'])['label']=='van'
    assert s.current(s.by_id['train:t1:0'])['status']=='reviewed'


@pytest.mark.parametrize('resize_other',[False,True])
def test_removal_export_keeps_shifted_object_extent_and_other_track_frames(session,resize_other):
    import copy
    s=session
    source_dataset=review.RosaCombinedDataset(s.root,s.paths['train'],cfg=review.SparseConfig(architecture_version=5),
                                             calibration=s.calibration)
    ext=json.loads(s.extents.read_text())
    car_record=copy.deepcopy(ext['frames']['t1'][0]);car_record['gt']=source_dataset.boxes[0][1].tolist()
    car_record['bbox']=[40.,50.,120.,160.];ext['frames']['t1'].append(car_record)
    s.extents.write_text(json.dumps(ext));s=reopen(s)
    source_hashes={k:review.sha(path) for k,path in s.paths.items()};extent_hash=review.sha(s.extents)
    s.remove('train:t1:0')
    s.apply('train:t2:0','bus',dimensions=[9.,3.,3.5] if resize_other else None)
    out=s.root/'removed_release';report=s.export(out)
    train=review.load_annotations(out/'fisheye3dod_infos_train.pkl')
    val=review.load_annotations(out/'fisheye3dod_infos_val.pkl')
    assert len(train['data_list'])==2
    assert train['data_list'][0]['annos']['gt_names'].tolist()==['Car']
    assert val['data_list'][0]['annos']['gt_names'][0]=='Truck'
    np.testing.assert_array_equal(train['data_list'][0]['annos']['gt_boxes'][0],
                                  s.documents['train']['data_list'][0]['annos']['gt_boxes'][1])
    assert train['data_list'][0]['annos']['gt_ids'].tolist()==[107]
    revised=json.loads((out/'train_yolo_extents_v1.json').read_text())
    assert revised['frames']['t1']==[car_record]
    assert len(revised['frames']['t2'])==(0 if resize_other else 1)
    if not resize_other:assert revised['frames']['t2'][0]['gt'][7]==5.
    assert report['extents']['removed_object_records_dropped']==1
    assert report['extents']['geometry_edited_records_dropped']==int(resize_other)
    assert report['splits']['train']['removed_boxes']==1
    assert review.verify_export(s.root,out,s.calibration)['all_boxes_checked']==5
    assert {k:review.sha(path) for k,path in s.paths.items()}==source_hashes
    assert review.sha(s.extents)==extent_hash


def test_val_only_removal_keeps_train_and_yolo_and_frame(session):
    s=session;s.remove('val:v1:0');out=s.root/'val_removed'
    report=s.export(out)
    val=review.load_annotations(out/'fisheye3dod_infos_val.pkl')
    assert len(val['data_list'])==1 and val['data_list'][0]['annos']['gt_ids'].tolist()==[107]
    assert report['splits']['val']['removed_boxes']==1
    assert report['splits']['train']['removed_boxes']==0
    assert report['extents']['removed_object_records_dropped']==0
    assert report['extents']['records']==2


def test_export_retains_frame_when_last_annotation_is_removed(session):
    old=session
    s=review.ReviewSession(old.root,old.paths['train'],old.paths['val'],old.extents,
                           old.calibration,old.state_path,classes=review.CLASSES)
    s.remove('train:t1:0');s.remove('train:t1:1');out=s.root/'empty_frame'
    s.export(out)
    train=review.load_annotations(out/'fisheye3dod_infos_train.pkl')
    assert len(train['data_list'])==2
    assert train['data_list'][0]['annos']['gt_boxes'].shape==(0,7)
    assert review.verify_export(s.root,out,s.calibration)['all_boxes_checked']==4


def test_independent_release_audit_checks_removed_indices_and_yolo(session):
    from audit_manual_release import audit
    s=session;s.remove('train:t1:0');s.apply('train:t2:0','bus')
    out=s.root/'audited';s.export(out);source=s.root/'source';source.mkdir()
    for split,path in s.paths.items():
        (source/f'fisheye3dod_infos_{split}.pkl').write_bytes(path.read_bytes())
    (source/'train_yolo_extents_v1.json').write_bytes(s.extents.read_bytes())
    result=audit(s.root,out,source,s.calibration)
    assert result['status']=='passed' and result['splits']['train']['removed']==1
    assert result['yolo']['removed']==1 and result['yolo']['changed']==1


def test_audit_flags_vehicle_sized_pedestrian_after_class_review(session):
    from audit_manual_release import audit
    s=session;s.apply('train:t1:0','pedestrian');out=s.root/'pedestrian_flag';s.export(out)
    source=s.root/'source';source.mkdir()
    for split,path in s.paths.items():
        (source/f'fisheye3dod_infos_{split}.pkl').write_bytes(path.read_bytes())
    (source/'train_yolo_extents_v1.json').write_bytes(s.extents.read_bytes())
    result=audit(s.root,out,source,s.calibration)
    assert result['status']=='passed'
    assert any(row['class_name']=='pedestrian' and row['id']=='train:t1:0' for row in result['geometry_flags'])


def test_removing_last_objects_keeps_empty_frame_and_aligned_optional_fields(session):
    import copy
    doc=copy.deepcopy(session.documents['train'])
    ann=doc['data_list'][0]['annos']
    ann['scores']=np.array([.9,.8]);ann['flags']=[True,False];ann['extra']=(1,2)
    result=review.remove_frame_annotations(doc,{'t1':{0,1}})
    kept=result['data_list'][0]['annos']
    assert kept['gt_boxes'].shape==(0,7) and kept['gt_names'].shape==(0,)
    assert kept['scores'].shape==(0,) and kept['flags']==[] and kept['extra']==()
    assert len(result['data_list'])==2 and len(doc['data_list'][0]['annos']['gt_boxes'])==2


def test_uncertain_preserves_label_and_dimensions(session):
    s=session;s.apply('train:t1:0','car',dimensions=[4.,1.8,1.5])
    s.apply('train:t1:0',uncertain=True,note='Occluded; check LiDAR')
    d=s.decisions['train:t1:0']
    assert d['label']=='car' and d['dimensions_lwh']==[4.,1.8,1.5]
    assert d['status']=='uncertain'


@pytest.mark.parametrize('size',[[4.,0.,1.5],[float('nan'),2.,3.],[4.,1.8]])
def test_invalid_size_is_rejected_without_mutation(session,size):
    with pytest.raises(ValueError):session.apply('train:t1:0','car',dimensions=size)
    assert not session.decisions and not session.undo_stack


def test_progress_rejects_changed_source_and_tampered_decisions(session):
    s=session;s.apply('train:t1:0','car')
    saved=json.loads(s.state_path.read_text());saved['decisions']['train:t1:0']['label']='unknown'
    review.atomic_json(s.state_path,saved)
    with pytest.raises(ValueError,match='Unknown annotation/class'):reopen(s)
    s.save();original=s.paths['train'].read_bytes()
    s.paths['train'].write_bytes(original+b' ')
    with pytest.raises(ValueError,match='different annotations'):reopen(s)


def test_export_syncs_class_and_preserves_sources_and_unreviewed(session):
    s=session;source_hashes={k:review.sha(p) for k,p in s.paths.items()};ext_hash=review.sha(s.extents)
    s.apply('train:t1:0','bus');out=s.root/'release'
    report=s.export(out)
    train=review.load_annotations(out/'fisheye3dod_infos_train.pkl')
    assert train['data_list'][0]['annos']['gt_names'].tolist()==['Bus','Car']
    assert train['data_list'][1]['annos']['gt_names'].tolist()==['Van','Car']
    assert train['data_list'][0]['annos']['gt_mesh'][0]=='detzero.bus'
    ext=json.loads((out/'train_yolo_extents_v1.json').read_text())
    assert ext['frames']['t1'][0]['gt'][7]==5.
    assert ext['evidence']=='preserve me' and ext['frames']['t1'][0]['bbox']==[20.,30.,100.,150.]
    assert report['extents']['changed_gt_records']==1
    assert report['splits']['train']['all_boxes_checked']==4
    assert report['splits']['val']['all_boxes_checked']==2
    assert {k:review.sha(p) for k,p in s.paths.items()}==source_hashes
    assert review.sha(s.extents)==ext_hash
    assert review.verify_export(s.root,out,s.calibration)['status']=='passed'
    with pytest.raises(FileExistsError):s.export(out)


def test_server_verification_needs_no_old_extent_but_rejects_tampering(session):
    s=session;out=s.root/'release';s.export(out)
    s.extents.unlink()
    assert review.verify_export(s.root,out,s.calibration)['all_boxes_checked']==6
    ext=out/'train_yolo_extents_v1.json';ext.write_text(ext.read_text()+' ')
    with pytest.raises(ValueError,match='extent hash mismatch'):
        review.verify_export(s.root,out,s.calibration)


def test_size_edit_drops_only_stale_extent_and_preserves_centers_yaw(session):
    s=session;s.apply('train:t1:0','car',dimensions=[4.,1.8,1.5])
    out=s.root/'sized';report=s.export(out)
    ext=json.loads((out/'train_yolo_extents_v1.json').read_text())
    assert ext['frames']['t1']==[] and len(ext['frames']['t2'])==1
    assert report['extents']['geometry_edited_records_dropped']==1
    original=s.documents['train']['data_list'][0]['annos']['gt_boxes'][0]
    revised=review.load_annotations(out/'fisheye3dod_infos_train.pkl')['data_list'][0]['annos']['gt_boxes'][0]
    np.testing.assert_array_equal(original[[0,1,2,6]],revised[[0,1,2,6]])
    np.testing.assert_allclose(revised[3:6],[4.,1.8,1.5])


def test_export_failure_does_not_leave_partial_destination(session):
    s=session;s.apply('train:t1:0','car')
    s.extents.write_text('{}');out=s.root/'bad'
    with pytest.raises(ValueError,match='extents/calibration changed'):s.export(out)
    assert not out.exists()


def test_four_camera_render_and_missing_image_are_explicit(session):
    s=session;views=s.render('train:t1:0')
    assert [v[0] for v in views]==list(review.CAMERAS)
    assert all(preview.width>0 and full.size==(800,640) for _,preview,full in views)
    s.apply('train:t1:0','car',dimensions=[4.,1.8,1.5])
    assert len(s.render('train:t1:0',crop=False))==4
    s.source_image.cache_clear();(s.root/'images/shared.png').unlink()
    with pytest.raises(FileNotFoundError):s.render('train:t1:0')
