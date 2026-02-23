from django.shortcuts import render
from rest_framework.views import View
from models import EdgeEvent, EventSerializer
from dashboard.models import CustomUser
# Create your views here.

class EdgeEventProcessor():
    
    def __init__(event: EdgeEvent):
        pass
    
    def normalize_keypoints(keypoints: dict): 
        """
        Normalize the keypoints so that they are centered (as done in model tests folder)
        
        Returns: 
            Numpy Array 
        """
        pass 
    
    def run_inference(keypoints: dict): 
        """
        Run Custom Modal inference on the keypoints for each frame 
        Returns the most common category as an array [X,X,X,X]
        """
        pass 
    
    def rule_based_classification(data: dict, event: EdgeEvent): 
        """
        Run custom rule based classification based on inference, devices state, and metadata from the event. updates the action for the vent
        """
        pass 
    
    def pre_proccess_event(self, event): # HTTP Request or sumn  
        ''' Recieve an Event from the edge device, and save it. Run inference and calssification in the background separetly. 
        This is a celery tasks that runs in the background 
        '''
        pass 
        
    def post_process_event(data: dict, user: CustomUser ): 
        """
        Call the AI API for all of a users is_processed=False events and determin which one are anomolus based on their historic data. Update the is_processed field and set is_alert for the ones the agent flags. 
        
        This is a scheduled celery tasks that runs in the background every 2 hours  
        """
        
        historic_events = EdgeEvent.objects.filter(
            device__user=user,
            device__is_active=True,
            is_processed=True,
            timestamp__gt= None, #TODO:  In the last 7 days 
        )
        
        payload = EdgeEvent.objects.filter(
            device__user=user,
            device__is_active=True,
            is_processed=False,
        )
        
        # Serialize and call modal with Structured output, List of event IDS that are alerts 
        
        pass 
    
    
class EdgeEventView(): # Should be a DRF model Class based view for easy READ, Update, and Delete
    
    def create(request): 
        """
        Create the event and record for the db and call the processor in the background, return right away.
        """
        

        
        
    
    

    
    
    
    
    