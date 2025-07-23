import json
import uuid
import asyncio
from typing import Dict, Any
from datetime import datetime

from ..agents.supervisor_agent import SupervisorAgent
from ..agents.flight_agent import FlightAgent
from ..agents.hotel_agent import HotelAgent
from ..database.operations import DatabaseOperations
from ..database.models import ChatMessage
from ..auth.supabase_auth import SupabaseAuth

class MainHandler:
    """
    Main handler that routes requests to appropriate agents for the main travel conversation flow
    ENHANCED: Now optimized for Function URL with extended timeouts
    """
    
    def __init__(self):
        self.db = DatabaseOperations()
        self.auth = SupabaseAuth()
        self.supervisor_agent = SupervisorAgent()
        self.flight_agent = FlightAgent()
        self.hotel_agent = HotelAgent()
    
    def _extract_request_body(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract request body from either API Gateway or Function URL event
        ENHANCED: Handles both event types
        """
        
        body = event.get('body')
        
        # Handle different event formats
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {}
        elif isinstance(body, dict):
            return body
        else:
            return {}
    
    async def handle_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Main handler for travel agent requests
        ENHANCED: Optimized for extended timeouts with Function URLs
        
        Args:
            event: Lambda event containing request data (API Gateway or Function URL)
            context: Lambda context
            
        Returns:
            Response dictionary
        """
        
        try:
            # Parse the incoming request (works for both API Gateway and Function URL)
            body = self._extract_request_body(event)
            
            # Extract key information
            user_id = body.get('user_id')
            message = body.get('message')
            session_id = body.get('session_id')  # Can be None
            access_token = body.get('access_token')
            
            # Detect event type for logging (should always be Function URL now)
            is_function_url = event.get('version') == '2.0' and 'rawPath' in event
            if not is_function_url:
                print(f"⚠️ WARNING: Travel chat received non-Function URL request - this is unexpected")
            
            event_type = "Function URL" if is_function_url else "API Gateway (Unexpected)"
            
            print(f"🔍 {event_type} request received - user_id: {user_id}, session_id: {session_id}, message: {message[:50] if message else 'None'}...")
            
            if not user_id or not message:
                return self._error_response(400, "Missing user_id or message")
            
            if not access_token:
                return self._error_response(401, "Missing access_token")
            
            # ENHANCED: Ensure session_id is always a valid UUID string
            if not session_id:
                session_id = str(uuid.uuid4())
                print(f"📝 Generated new session_id: {session_id}")
            else:
                # Validate existing session_id
                try:
                    uuid.UUID(session_id)  # Validate it's a proper UUID
                    print(f"✅ Using existing session_id: {session_id}")
                except ValueError:
                    session_id = str(uuid.uuid4())
                    print(f"⚠️ Invalid session_id provided, generated new one: {session_id}")
            
            # Verify user authentication
            print(f"🔐 Verifying authentication...")
            auth_result = await self.auth.verify_token(access_token)
            if not auth_result.get("success"):
                return self._error_response(401, "Invalid or expired token")
            
            print(f"✅ Authentication verified")
            
            # Get user's profile and conversation history
            print(f"👤 Getting user profile...")
            user_profile = await self.db.get_user_profile(user_id)
            if not user_profile:
                return self._error_response(404, "User profile not found")
            
            print(f"👤 User profile found: {user_profile.email}")
            
            # ENHANCED: Better error handling for conversation history
            try:
                print(f"💬 Getting conversation history...")
                conversation_history = await self.db.get_conversation_history(user_id, session_id)
                print(f"💬 Retrieved {len(conversation_history)} previous messages")
            except Exception as e:
                print(f"⚠️ Error getting conversation history: {e}, continuing with empty history")
                conversation_history = []
            
            # ENHANCED: Process message through supervisor agent with extended timeout support
            print(f"🤖 Processing with SupervisorAgent...")
            start_time = datetime.now()
            
            supervisor_response = await self.supervisor_agent.process_message(
                user_id=user_id,
                message=message,
                session_id=session_id,
                user_profile=user_profile.model_dump(),
                conversation_history=conversation_history
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"🤖 Supervisor response completed in {processing_time:.2f}s - agent_type: {supervisor_response.get('agent_type')}")
            
            # Check if we need to route to a specific agent
            if supervisor_response.get("requires_routing"):
                route_to = supervisor_response.get("route_to")
                
                print(f"🔀 Routing to specialized agent: {route_to}")
                
                if route_to == "flight_agent":
                    agent_response = await self.flight_agent.process_message(
                        user_id=user_id,
                        message=message,
                        session_id=session_id,
                        user_profile=user_profile.model_dump(),
                        conversation_history=conversation_history
                    )
                elif route_to == "hotel_agent":
                    agent_response = await self.hotel_agent.process_message(
                        user_id=user_id,
                        message=message,
                        session_id=session_id,
                        user_profile=user_profile.model_dump(),
                        conversation_history=conversation_history
                    )
                else:
                    # Fallback to supervisor for general conversation
                    agent_response = await self.supervisor_agent.handle_general_conversation(
                        user_id=user_id,
                        message=message,
                        session_id=session_id,
                        user_profile=user_profile.model_dump(),
                        conversation_history=conversation_history
                    )
            else:
                # Use supervisor response directly
                agent_response = supervisor_response
            
            total_processing_time = (datetime.now() - start_time).total_seconds()
            print(f"⚡ Total processing time: {total_processing_time:.2f}s")
            
            # ENHANCED: Ensure session_id is not None before saving
            if not session_id:
                session_id = str(uuid.uuid4())
                print(f"⚠️ session_id was None before saving, generated: {session_id}")
            
            # Save conversation to database with better error handling (async for performance)
            try:
                print(f"💾 Saving conversation...")
                # Use asyncio.create_task for non-blocking save
                save_task = asyncio.create_task(
                    self.db.save_conversation_turn(
                        user_id=user_id,
                        session_id=session_id,
                        user_message=message,
                        assistant_response=agent_response.get('message', ''),
                        metadata=agent_response.get('metadata', {})
                    )
                )
                
                # Don't wait for save to complete (fire-and-forget for better performance)
                print("💾 Conversation save initiated (async)")
            except Exception as e:
                print(f"⚠️ Error initiating conversation save: {e}, continuing with response")
                # Don't fail the request if saving fails
            
            # Format response
            response_data = {
                'response': agent_response.get('message'),
                'agent_type': agent_response.get('agent_type', 'main_travel'),
                'session_id': session_id,  # Always return the session_id
                'requires_action': agent_response.get('requires_action', False),
                'processing_time': total_processing_time,
                'event_type': event_type
            }
            
            # Add additional data based on agent type
            if agent_response.get('agent_type') == 'flight':
                response_data.update({
                    'search_results': agent_response.get('search_results', []),
                    'search_id': agent_response.get('search_id'),
                    'flight_selected': agent_response.get('flight_selected'),
                    'filtering_info': agent_response.get('filtering_info', {})
                })
            elif agent_response.get('agent_type') == 'hotel':
                response_data.update({
                    'search_results': agent_response.get('search_results', []),
                    'search_id': agent_response.get('search_id'),
                    'hotel_selected': agent_response.get('hotel_selected'),
                    'filtering_info': agent_response.get('filtering_info', {})
                })
            
            # Add metadata if present
            if agent_response.get('metadata'):
                response_data['metadata'] = agent_response['metadata']
            
            print(f"✅ Request completed successfully in {total_processing_time:.2f}s via {event_type}")
            return self._success_response(response_data)
            
        except Exception as e:
            print(f"❌ Error in main handler: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._error_response(500, "Internal server error")
    
    async def handle_auth_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Handle authentication requests (sign up, sign in, etc.) - ENHANCED VERSION WITH ROBUST PROFILE CREATION
        
        Args:
            event: Lambda event containing auth request
            context: Lambda context
            
        Returns:
            Authentication response
        """
        
        try:
            body = self._extract_request_body(event)
            action = body.get('action')  # 'signup', 'signin', 'refresh', 'signout'
            
            if action == 'signup':
                email = body.get('email')
                password = body.get('password')
                metadata = body.get('metadata', {})
                
                if not email or not password:
                    return self._error_response(400, "Missing email or password")
                
                auth_result = await self.auth.sign_up(email, password, metadata)
                
                if auth_result.get("success"):
                    # ENHANCED: More robust user profile creation with enhanced retry logic
                    user_created_id = auth_result["user"]["id"]
                    print(f"🎉 User created with ID: {user_created_id}")
                    
                    try:
                        # ENHANCED: Enhanced profile creation with multiple verification steps
                        profile_created = False
                        max_attempts = 5  # Increased attempts
                        
                        for attempt in range(max_attempts):
                            try:
                                print(f"🔄 Profile creation attempt {attempt + 1} for user: {user_created_id}")
                                
                                # Check if user profile already exists
                                existing_profile = await self.db.get_user_profile(user_created_id)
                                
                                if existing_profile:
                                    print(f"ℹ️ User profile already exists for: {email}")
                                    # Verify the profile is properly set up
                                    if existing_profile.email == email:
                                        print(f"✅ Existing profile verified for: {email}")
                                        profile_created = True
                                        break
                                    else:
                                        print(f"⚠️ Existing profile has wrong email, updating...")
                                        # Update the existing profile with correct email
                                        update_success = await self.db.update_user_profile(user_created_id, {"email": email})
                                        if update_success:
                                            print(f"✅ Profile email updated successfully")
                                            profile_created = True
                                            break
                                
                                # Create initial user profile
                                from ..database.models import UserProfile
                                
                                user_profile = UserProfile(
                                    user_id=user_created_id,
                                    email=email,
                                    onboarding_completed=False,
                                    is_active=True
                                )
                                
                                profile_created = await self.db.create_user_profile(user_profile)
                                
                                if profile_created:
                                    print(f"✅ User profile created successfully for: {email}")
                                    
                                    # ENHANCED: Multiple verification attempts with delays
                                    verification_success = False
                                    for verify_attempt in range(3):
                                        await asyncio.sleep(0.1 * (verify_attempt + 1))  # Progressive delay
                                        verification_profile = await self.db.get_user_profile(user_created_id)
                                        if verification_profile and verification_profile.email == email:
                                            print(f"✅ Profile creation verified for: {email} (attempt {verify_attempt + 1})")
                                            verification_success = True
                                            break
                                        else:
                                            print(f"⚠️ Profile verification failed, attempt {verify_attempt + 1}")
                                    
                                    if verification_success:
                                        profile_created = True
                                        break
                                    else:
                                        print(f"⚠️ Profile creation verification failed after 3 attempts, retrying creation...")
                                        profile_created = False
                                else:
                                    print(f"⚠️ Profile creation attempt {attempt + 1} failed for: {email}")
                                    
                            except Exception as attempt_error:
                                print(f"⚠️ Profile creation attempt {attempt + 1} error: {attempt_error}")
                                import traceback
                                traceback.print_exc()
                                continue
                        
                        if not profile_created:
                            print(f"❌ Failed to create profile after {max_attempts} attempts for: {email}")
                            # ENHANCED: Still return success for auth, but log the profile creation failure
                            print(f"⚠️ User authentication succeeded but profile creation failed")
                            print(f"⚠️ Profile will be created during onboarding if needed")
                        else:
                            print(f"✅ Profile creation process completed successfully for: {email}")
                        
                    except Exception as profile_error:
                        # ENHANCED: Don't fail signup if profile creation fails, but log it
                        print(f"⚠️ User profile creation error (non-critical): {profile_error}")
                        import traceback
                        traceback.print_exc()
                        print(f"⚠️ Profile will be created during onboarding if needed")
                    
                    # Convert datetime objects to strings for JSON serialization
                    user_data = auth_result["user"].copy()
                    if isinstance(user_data.get("created_at"), datetime):
                        user_data["created_at"] = user_data["created_at"].isoformat()
                    if user_data.get("updated_at"):
                        if isinstance(user_data["updated_at"], datetime):
                            user_data["updated_at"] = user_data["updated_at"].isoformat()
                    if user_data.get("email_confirmed_at"):
                        if isinstance(user_data["email_confirmed_at"], datetime):
                            user_data["email_confirmed_at"] = user_data["email_confirmed_at"].isoformat()
                    if user_data.get("last_sign_in_at"):
                        if isinstance(user_data["last_sign_in_at"], datetime):
                            user_data["last_sign_in_at"] = user_data["last_sign_in_at"].isoformat()
                    
                    session_data = auth_result["session"].copy()
                    if isinstance(session_data.get("expires_at"), datetime):
                        session_data["expires_at"] = session_data["expires_at"].isoformat()
                    elif isinstance(session_data.get("expires_at"), (int, float)):
                        # If it's already a timestamp, convert to ISO format
                        try:
                            session_data["expires_at"] = datetime.fromtimestamp(session_data["expires_at"]).isoformat()
                        except (ValueError, TypeError):
                            # If conversion fails, keep the original value
                            pass
                    
                    return self._success_response({
                        'user': user_data,
                        'session': session_data,
                        'message': 'User created successfully'
                    })
                else:
                    return self._error_response(400, auth_result.get("error", "Signup failed"))
            
            elif action == 'signin':
                email = body.get('email')
                password = body.get('password')
                
                if not email or not password:
                    return self._error_response(400, "Missing email or password")
                
                auth_result = await self.auth.sign_in(email, password)
                
                if auth_result.get("success"):
                    # Convert datetime objects to strings for JSON serialization
                    user_data = auth_result["user"].copy()
                    if isinstance(user_data.get("last_sign_in_at"), datetime):
                        user_data["last_sign_in_at"] = user_data["last_sign_in_at"].isoformat()
                    if user_data.get("email_confirmed_at"):
                        if isinstance(user_data["email_confirmed_at"], datetime):
                            user_data["email_confirmed_at"] = user_data["email_confirmed_at"].isoformat()
                    if user_data.get("created_at"):
                        if isinstance(user_data["created_at"], datetime):
                            user_data["created_at"] = user_data["created_at"].isoformat()
                    
                    session_data = auth_result["session"].copy()
                    if isinstance(session_data.get("expires_at"), datetime):
                        session_data["expires_at"] = session_data["expires_at"].isoformat()
                    elif isinstance(session_data.get("expires_at"), (int, float)):
                        # If it's already a timestamp, convert to ISO format
                        try:
                            session_data["expires_at"] = datetime.fromtimestamp(session_data["expires_at"]).isoformat()
                        except (ValueError, TypeError):
                            # If conversion fails, keep the original value
                            pass
                    
                    return self._success_response({
                        'user': user_data,
                        'session': session_data,
                        'message': 'Signed in successfully'
                    })
                else:
                    return self._error_response(401, auth_result.get("error", "Sign in failed"))
            
            elif action == 'refresh':
                refresh_token = body.get('refresh_token')
                
                if not refresh_token:
                    return self._error_response(400, "Missing refresh_token")
                
                auth_result = await self.auth.refresh_session(refresh_token)
                
                if auth_result.get("success"):
                    # Convert datetime objects to strings for JSON serialization
                    session_data = auth_result["session"].copy()
                    if isinstance(session_data.get("expires_at"), datetime):
                        session_data["expires_at"] = session_data["expires_at"].isoformat()
                    elif isinstance(session_data.get("expires_at"), (int, float)):
                        try:
                            session_data["expires_at"] = datetime.fromtimestamp(session_data["expires_at"]).isoformat()
                        except (ValueError, TypeError):
                            pass
                    
                    response_data = {
                        'session': session_data,
                        'message': 'Session refreshed successfully'
                    }
                    
                    # Add user data if available
                    if auth_result.get("user"):
                        user_data = auth_result["user"].copy()
                        if isinstance(user_data.get("email_confirmed_at"), datetime):
                            user_data["email_confirmed_at"] = user_data["email_confirmed_at"].isoformat()
                        response_data['user'] = user_data
                    
                    return self._success_response(response_data)
                else:
                    return self._error_response(401, auth_result.get("error", "Token refresh failed"))
            
            elif action == 'signout':
                auth_result = await self.auth.sign_out()
                return self._success_response({
                    'message': 'Signed out successfully'
                })
            
            else:
                return self._error_response(400, f"Unknown action: {action}")
                
        except Exception as e:
            print(f"Error in auth handler: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._error_response(500, "Internal server error")
    
    async def handle_user_status(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Handle user status requests with ENHANCED debugging and error recovery
        """
        
        try:
            body = self._extract_request_body(event)
            user_id = body.get('user_id')
            access_token = body.get('access_token')
            
            print(f"🔍 ENHANCED DEBUG: User status request for user_id: {user_id}")
            
            if not user_id or not access_token:
                print(f"❌ Missing required fields: user_id={bool(user_id)}, access_token={bool(access_token)}")
                return self._error_response(400, "Missing user_id or access_token")
            
            # Verify authentication
            print(f"🔐 Verifying access token...")
            auth_result = await self.auth.verify_token(access_token)
            if not auth_result.get("success"):
                print(f"❌ Token verification failed: {auth_result.get('error')}")
                return self._error_response(401, "Invalid or expired token")
            
            print(f"✅ Token verified successfully")
            
            # ENHANCED: ENHANCED debugging and error recovery for user status
            print(f"🔍 Getting user status with enhanced debugging...")
            
            try:
                # STEP 1: Direct profile check
                print(f"📊 STEP 1: Checking user profile directly...")
                user_profile = await self.db.get_user_profile(user_id)
                
                if user_profile:
                    print(f"✅ User profile found: email={user_profile.email}, onboarding_completed={user_profile.onboarding_completed}")
                    
                    # STEP 2: Try get_user_status
                    print(f"📊 STEP 2: Trying get_user_status...")
                    user_status = await self.db.get_user_status(user_id)
                    
                    if user_status:
                        print(f"✅ User status retrieved successfully")
                        print(f"📊 Status details: onboarding_completed={user_status.get('onboarding_completed')}")
                        return self._success_response(user_status)
                    else:
                        print(f"⚠️ get_user_status returned None, but profile exists")
                        print(f"🔧 Creating status from profile directly...")
                        
                        # STEP 3: Manual status creation from profile
                        try:
                            # Get onboarding progress
                            onboarding = await self.db.get_onboarding_progress(user_id)
                            print(f"📈 Onboarding progress: {onboarding.current_step.value if onboarding else 'None'}")
                            
                            # Get user groups
                            groups = await self.db.get_user_groups(user_id)
                            print(f"👥 User groups: {len(groups)} groups")
                            
                            # Create manual status
                            manual_status = {
                                "user_id": user_id,
                                "email": user_profile.email,
                                "onboarding_completed": user_profile.onboarding_completed,
                                "is_active": user_profile.is_active,
                                "current_step": onboarding.current_step.value if onboarding else "group_code",
                                "completion_percentage": onboarding.completion_percentage if onboarding else (100.0 if user_profile.onboarding_completed else 0.0),
                                "groups": groups,
                                "profile": user_profile.dict()
                            }
                            
                            print(f"✅ Manual status created successfully")
                            return self._success_response(manual_status)
                            
                        except Exception as manual_error:
                            print(f"❌ Manual status creation failed: {manual_error}")
                            import traceback
                            traceback.print_exc()
                            
                            # STEP 4: Minimal fallback status
                            print(f"🔧 Creating minimal fallback status...")
                            fallback_status = {
                                "user_id": user_id,
                                "email": user_profile.email,
                                "onboarding_completed": user_profile.onboarding_completed,
                                "is_active": user_profile.is_active,
                                "current_step": "unknown",
                                "completion_percentage": 100.0 if user_profile.onboarding_completed else 0.0,
                                "groups": [],
                                "profile": user_profile.dict()
                            }
                            
                            print(f"✅ Fallback status created")
                            return self._success_response(fallback_status)
                else:
                    print(f"❌ No user profile found for user_id: {user_id}")
                    
                    # STEP 5: Check if this is a database issue or missing profile
                    print(f"🔍 STEP 5: Investigating missing profile...")
                    
                    # Check if user exists in auth but not in our database
                    auth_user = auth_result.get("user")
                    if auth_user:
                        print(f"🔍 Auth user exists: {auth_user.get('email')}")
                        print(f"❌ Profile missing from database - this indicates a signup issue")
                        
                        # Try to create missing profile from auth data
                        try:
                            print(f"🔧 Attempting to create missing profile from auth data...")
                            from ..database.models import UserProfile
                            
                            missing_profile = UserProfile(
                                user_id=user_id,
                                email=auth_user.get("email"),
                                onboarding_completed=False,
                                is_active=True
                            )
                            
                            profile_created = await self.db.create_user_profile(missing_profile)
                            
                            if profile_created:
                                print(f"✅ Missing profile created successfully")
                                
                                # Return basic status for newly created profile
                                recovery_status = {
                                    "user_id": user_id,
                                    "email": auth_user.get("email"),
                                    "onboarding_completed": False,
                                    "is_active": True,
                                    "current_step": "group_code",
                                    "completion_percentage": 0.0,
                                    "groups": [],
                                    "profile": missing_profile.dict()
                                }
                                
                                print(f"✅ Recovery status created")
                                return self._success_response(recovery_status)
                            else:
                                print(f"❌ Failed to create missing profile")
                                
                        except Exception as recovery_error:
                            print(f"❌ Profile recovery failed: {recovery_error}")
                            import traceback
                            traceback.print_exc()
                    
                    return self._error_response(404, "User not found")
                    
            except Exception as status_error:
                print(f"❌ Error during user status retrieval: {status_error}")
                import traceback
                traceback.print_exc()
                return self._error_response(500, f"Error retrieving user status: {str(status_error)}")
                
        except Exception as e:
            print(f"❌ Error in user status handler: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._error_response(500, "Internal server error")
    
    def _success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return successful API response with proper JSON serialization"""
        
        def json_serializer(obj):
            """JSON serializer for objects not serializable by default json code"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'dict'):  # Pydantic models
                return obj.dict()
            elif hasattr(obj, '__dict__'):  # Other objects
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({
                'success': True,
                'data': data,
                'timestamp': datetime.utcnow().isoformat()
            }, default=json_serializer)
        }
    
    def _error_response(self, status_code: int, error_message: str) -> Dict[str, Any]:
        """Return error API response"""
        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': error_message,
                'timestamp': datetime.utcnow().isoformat()
            })
        }