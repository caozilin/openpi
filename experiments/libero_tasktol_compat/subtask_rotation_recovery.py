"""Causal SO(3) action recovery between released manipulation cycles.

This is an action/phase callback; all solving and native execution stay in the
shared session. Original nominal rotation is tracked independently throughout.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.spatial.transform import Rotation


def exp(v):return Rotation.from_rotvec(v).as_matrix()
def log(r):return Rotation.from_matrix(r).as_rotvec()
def smoothstep(x):return x*x*(3.-2.*x)

@dataclass(frozen=True)
class RecoverySettings:
    dt_s: float=.05
    correction_speed_deg_s: float=45.
    combined_speed_deg_s: float=45.
    released_aperture_m: float=.07
    completion_threshold_deg: float=.5
    warning_after_s: float=2.

    def __post_init__(self):
        if min(self.dt_s,self.correction_speed_deg_s,self.combined_speed_deg_s,self.released_aperture_m,
               self.completion_threshold_deg,self.warning_after_s)<=0:raise ValueError('Recovery settings must be positive')

class RotationRecovery:
    def __init__(self,anchor_rotation,eligible_cycles,settings=RecoverySettings()):
        self.original_rotation=np.asarray(anchor_rotation).copy()
        self.eligible_cycles=set(eligible_cycles)
        self.settings=settings;self.handled=set();self.active=None;self.events=[]
        self.had_execution=False;self.index=0

    def start(self,index,cycle,accepted):
        residual=log(self.original_rotation.T@accepted)
        angle=float(np.linalg.norm(residual))
        limit=np.deg2rad(self.settings.correction_speed_deg_s)*self.settings.dt_s
        # max d/dx smoothstep=1.5, so this duration bounds the extra SO(3) step.
        frames=max(1,int(np.ceil(1.5*angle/limit)))
        event=dict(start_frame=index,cycle=cycle,initial_offset_deg=float(np.degrees(angle)),
            planned_frames=frames,minimum_duration_s=frames*self.settings.dt_s,completed=False,
            applied_frames=0,speed_limited_frames=0,source_over_speed_frames=0)
        self.events.append(event)
        self.active=dict(residual=residual,progress=0.,frames=frames,event=event)
        self.handled.add(cycle)

    def advance(self,base_action,current_nominal,*,allow_recovery=True):
        action=np.asarray(base_action).copy();u=exp(action[3:6])
        desired_original=u@self.original_rotation
        details=dict(active=False,correction_angle_deg=0.,progress=None)
        if self.active is not None and allow_recovery:
            a=self.active;x=a['progress'];proposed=min(1.,x+1./a['frames'])
            cap=np.deg2rad(self.settings.combined_speed_deg_s)*self.settings.dt_s
            def candidate(progress):
                desired=desired_original@exp((1.-smoothstep(progress))*a['residual'])
                increment=desired@current_nominal.T
                return increment,desired
            # A pre-existing oversized source action is preserved, never
            # clamped or retimed; no extra recovery is added on such a frame.
            source_angle=float(np.linalg.norm(log(u)))
            if source_angle>cap+1e-10:
                proposed=x;a['event']['source_over_speed_frames']+=1
            else:
                increment,_=candidate(proposed)
                if np.linalg.norm(log(increment))>cap+1e-10:
                    lo,hi=x,proposed
                    for _ in range(40):
                        mid=(lo+hi)/2
                        if np.linalg.norm(log(candidate(mid)[0]))<=cap:lo=mid
                        else:hi=mid
                    proposed=lo;a['event']['speed_limited_frames']+=1
            increment,desired=candidate(proposed)
            if source_angle>cap+1e-10:increment=u;desired=u@current_nominal
            action[3:6]=log(increment)
            correction=log(increment@u.T)
            assert np.linalg.norm(correction)<=np.deg2rad(self.settings.correction_speed_deg_s)*self.settings.dt_s+1e-8
            a['progress']=proposed;a['event']['applied_frames']+=1
            details=dict(active=True,correction_angle_deg=float(np.degrees(np.linalg.norm(correction))),
                correction_world_rotvec_rad=correction.tolist(),progress=proposed,
                carrier_remaining_offset_deg=float(np.degrees(np.linalg.norm(log(desired_original@desired.T)))),
                combined_action_angle_deg=float(np.degrees(np.linalg.norm(action[3:6]))),
                source_action_angle_deg=float(np.degrees(source_angle)))
            details['elapsed_s']=a['event']['applied_frames']*self.settings.dt_s
            details['within_reference_threshold']=details['carrier_remaining_offset_deg']<=self.settings.completion_threshold_deg
            details['over_warning_time']=details['elapsed_s']>self.settings.warning_after_s
            a['event']['elapsed_s']=details['elapsed_s'];a['event']['over_warning_time']=details['over_warning_time']
            if proposed>=1.-1e-12:
                a['event'].update(completed=True,end_frame=self.index,residual_deg=details['carrier_remaining_offset_deg'])
                self.active=None
        self.original_rotation=desired_original;self.index+=1
        return action,details

    def __call__(self,index,base_action,phase,phase_details,session):
        from tasktol.tolerance.runtime import rebase_rotation_subtask
        assert index==self.index
        info=dict(phase_details);cycle=phase.subtask_index;opening=not bool(base_action[6]>0)
        if phase.phase.key=='postgrasp':self.had_execution=True
        allowed=opening and phase.phase.key=='pregrasp'
        ready=allowed and phase.aperture_m>=self.settings.released_aperture_m
        if self.active is not None and not allowed:
            self.active['event'].update(end_frame=index,interrupted_by=phase.phase.key,
                residual_deg=float(np.degrees(np.linalg.norm(log(self.original_rotation@session.release.accepted_rotation.T)))))
            self.active=None
        if self.active is None and ready and self.had_execution and cycle in self.eligible_cycles and cycle not in self.handled:
            # Reset the chart around the accepted command only. No actual pose
            # or joint teleport occurs, and the independent original is kept.
            accepted=session.release.accepted_rotation.copy()
            key=(cycle,info['phase'])
            rebase_rotation_subtask(session.target,session.tolerance,session.release,stage_key=key,accepted_rotation=accepted)
            session.stage_key=key;session.subtask_index=cycle
            self.start(index,cycle,accepted)
            info['recovery_reference_rebased']=True
        was_active=self.active is not None
        before_error=float(np.degrees(np.linalg.norm(log(self.original_rotation@session.release.accepted_rotation.T))))
        action,diagnostic=self.advance(base_action,session.target.rotation,allow_recovery=allowed)
        diagnostic['accepted_original_error_before_deg']=before_error
        if was_active:
            info['authored_semantic_mask']=info['semantic_mask'];info['semantic_mask']='000';info['mask_prediction']='000'
        info.update(base_action=np.asarray(base_action).tolist(),rotation_recovery=diagnostic,
            original_nominal_rotation_vector_rad=log(self.original_rotation).tolist())
        return action,info

    def observe(self,row,session):
        """Record executed results only; never feed actual tracking error back."""
        diagnostic=row['rotation_recovery']
        actual=exp(row['post_action']['rotation_vector_rad'])
        accepted=session.release.accepted_rotation
        diagnostic['accepted_original_error_after_deg']=float(np.degrees(np.linalg.norm(log(self.original_rotation@accepted.T))))
        diagnostic['actual_original_error_after_deg']=float(np.degrees(np.linalg.norm(log(self.original_rotation@actual.T))))
        if row['aborted'] and self.events and diagnostic['active']:
            self.events[-1].update(completed=False,rejected_during_recovery=True,
                accepted_residual_at_stop_deg=diagnostic['accepted_original_error_after_deg'])
        if row['observed_phase']=='grasp':
            event=next((e for e in reversed(self.events)if e['cycle']==row['observed_subtask_index']),None)
            if event is not None and 'next_grasp_frame'not in event:
                event.update(next_grasp_frame=row['index'],
                    next_grasp_accepted_residual_deg=diagnostic['accepted_original_error_before_deg'])

    def report(self):return dict(settings=asdict(self.settings),events=self.events,
        complete=sum(e['completed']for e in self.events),total=len(self.events),unfinished_at_end=self.active is not None)
