"""Strict two-object Windows launcher tree; no queries or side effects."""
def require(ok,message):
    if not ok:raise ValueError(message)

def capture(items,*,launcher_pid,controller_pid,module,nonce):
    primary=[r for r in items if r['pid']==launcher_pid]
    require(len(primary)==1,'Exact Popen native launcher required')
    parent=primary[0]
    require(parent['parent']==controller_pid and module in parent['command'] and nonce in parent['command'],
            'Native launcher parent/module/admission differs')
    children=[r for r in items if r['parent']==launcher_pid]
    require(len(children)<=1,'Ambiguous native launcher shim tree')
    for child in children:
        require(child['command']==parent['command'] and child['pid']!=parent['pid']
                and type(child['birth']) is str and child['birth'].isdigit(),
                'Native shim command or birth differs')
    require(type(parent['birth']) is str and parent['birth'].isdigit(),'Exact launcher birth required')
    return [parent]+children

def verify(items,binding,*,launcher_pid,controller_pid,module,nonce):
    observed=capture(items,launcher_pid=launcher_pid,controller_pid=controller_pid,module=module,nonce=nonce)
    require(observed==binding,'Captured launcher/shim birth/parent/command changed')
    return observed
