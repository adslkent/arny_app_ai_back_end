import json
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from openai import OpenAI
from agents import Agent, Runner, function_tool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception_message,
    retry_any,
    retry_if_result,
    before_sleep_log,
    retry_if_exception
)
from pydantic import BaseModel, ValidationError
import logging
import requests

from ..utils.config import config
from ..utils.group_codes import GroupCodeGenerator
from ..services.email_service import EmailService
from ..database.operations import DatabaseOperations
from ..database.models import OnboardingStep, OnboardingProgress

# Set up logging
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS FOR VALIDATION ====================

class OpenAIResponse(BaseModel):
    """Pydantic model for OpenAI response validation"""
    output: Optional[Any] = None

class AgentRunnerResponse(BaseModel):
    """Pydantic model for Agent Runner response validation"""
    final_output: Optional[str] = None

# ==================== OPENAI API RETRY CONDITIONS ====================

def retry_on_openai_api_error_result(result):
    """Condition 2: Retry if OpenAI API result contains error/warning fields"""
    if hasattr(result, 'error') and result.error:
        return True
    if isinstance(result, dict):
        return (
            result.get("error") is not None or
            "warning" in result or
            result.get("success") is False
        )
    return False

def retry_on_openai_http_status_error(result):
    """Condition 1: Retry on HTTP status errors for OpenAI API calls"""
    if hasattr(result, 'status_code'):
        return result.status_code >= 400
    if hasattr(result, 'response') and hasattr(result.response, 'status_code'):
        return result.response.status_code >= 400
    return False

def retry_on_openai_validation_failure(result):
    """Condition 5: Retry if OpenAI result fails Pydantic validation"""
    try:
        if result:
            OpenAIResponse.model_validate(result.__dict__ if hasattr(result, '__dict__') else result)
            return False  # Validation passed
        return True  # Empty result
    except ValidationError:
        return True  # Validation failed

def retry_on_agent_runner_validation_failure(result):
    """Condition 5: Retry if Agent Runner result fails Pydantic validation"""
    try:
        if result:
            AgentRunnerResponse.model_validate(result.__dict__ if hasattr(result, '__dict__') else result)
            return False  # Validation passed
        return True  # Empty result
    except ValidationError:
        return True  # Validation failed

def retry_on_openai_api_exception(exception):
    """Condition 4: Custom exception checker for OpenAI API calls"""
    exception_str = str(exception).lower()
    return any(keyword in exception_str for keyword in [
        'timeout', 'failed', 'unavailable', 'rate limit', 'api error',
        'connection', 'network', 'server error'
    ])

# ==================== RETRY DECORATORS ====================

openai_responses_api_retry = retry(
    reraise=True,
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|rate.limit|api.error|connection|network|server.error).*"),
        # Condition 4: Exception types and custom checkers
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        retry_if_exception(retry_on_openai_api_exception),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_openai_api_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_openai_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_openai_validation_failure)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

openai_agents_sdk_retry = retry(
    reraise=True,
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|rate.limit|api.error|connection|network|server.error).*"),
        # Condition 4: Exception types and custom checkers
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        retry_if_exception(retry_on_openai_api_exception),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_openai_api_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_openai_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_agent_runner_validation_failure)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# Global variable to store the current agent instance
_current_onboarding_agent = None

def _get_onboarding_agent():
    """Get the current onboarding agent instance"""
    global _current_onboarding_agent
    return _current_onboarding_agent

def _run_async_safely(coro):
    """Run async coroutine safely by using the current event loop or creating a new one"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_in_new_loop, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)

def _run_in_new_loop(coro):
    """Run coroutine in a completely new event loop"""
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()
        asyncio.set_event_loop(None)

# ==================== ENHANCED TOOL FUNCTIONS ====================

@function_tool
def scan_email_for_profile_tool(email: str) -> dict:
    """
    Scan email for profile information using environment-appropriate method
    Returns a dict with keys: name, gender, birthdate, city
    """
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"📧 Scanning email for profile: {email}")
        
        # CRITICAL: Always store the email address first
        agent.current_collected_data["email"] = email
        print(f"✅ Email stored: {email}")
        
        # Try to scan the email for profile information
        try:
            # Use the email service to scan for profile data
            scan_result = _run_async_safely(agent.email_service.scan_email_for_profile(email, agent.current_user_id))
            
            print(f"📧 Email scan result: {scan_result}")
            
            if scan_result.get("success") and scan_result.get("name"):
                # Successfully found some profile information
                extracted_info = []
                
                # Store any found information
                if scan_result.get("name"):
                    agent.current_collected_data["scanned_name"] = scan_result["name"]
                    extracted_info.append(f"Name: {scan_result['name']}")
                
                if scan_result.get("gender"):
                    agent.current_collected_data["scanned_gender"] = scan_result["gender"]
                    extracted_info.append(f"Gender: {scan_result['gender']}")
                
                if scan_result.get("birthdate"):
                    agent.current_collected_data["scanned_birthdate"] = scan_result["birthdate"]
                    extracted_info.append(f"Birthdate: {scan_result['birthdate']}")
                
                if scan_result.get("city"):
                    agent.current_collected_data["scanned_city"] = scan_result["city"]
                    extracted_info.append(f"City: {scan_result['city']}")
                
                if extracted_info:
                    return {
                        "success": True,
                        "email_stored": email,
                        "extracted_info": extracted_info,
                        "message": f"Great! I found some information from your email: {', '.join(extracted_info)}. Please confirm if this is correct and provide any missing details."
                    }
                else:
                    return {
                        "success": True,
                        "email_stored": email,
                        "message": f"I've connected your email ({email}) but couldn't extract profile information automatically. I'll help you fill out your profile manually instead."
                    }
            else:
                # Scanning failed or no information found
                print(f"⚠️ Email scanning failed or no info found: {scan_result.get('error', 'Unknown error')}")
                return {
                    "success": True,
                    "email_stored": email,
                    "message": f"I've connected your email ({email}). Email scanning is currently not available, so I'll help you fill out your profile manually instead."
                }
                
        except Exception as scan_error:
            print(f"⚠️ Email scanning error: {scan_error}")
            return {
                "success": True,
                "email_stored": email,
                "message": f"I've connected your email ({email}). I'll help you fill out your profile manually."
            }
        
    except Exception as e:
        print(f"❌ Error in scan_email_for_profile_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def store_personal_info_tool(name: str, gender: str, birthdate: str, city: str) -> dict:
    """Store personal information"""
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"👤 Storing personal info: {name}, {gender}, {birthdate}, {city}")
        
        agent.current_collected_data.update({
            "name": name,
            "gender": gender,
            "birthdate": birthdate,
            "city": city
        })
        
        print(f"✅ Personal info stored successfully")
        return {"success": True, "message": "Personal information saved successfully."}
        
    except Exception as e:
        print(f"❌ Error in store_personal_info_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def store_job_details_tool(employer: str, working_schedule: str, holiday_frequency: str) -> dict:
    """Store job details"""
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"💼 Storing job details: {employer}, {working_schedule}, {holiday_frequency}")
        
        agent.current_collected_data.update({
            "employer": employer,
            "working_schedule": working_schedule,
            "holiday_frequency": holiday_frequency
        })
        
        print(f"✅ Job details stored successfully")
        return {"success": True, "message": "Job details saved successfully."}
        
    except Exception as e:
        print(f"❌ Error in store_job_details_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def store_financial_info_tool(annual_income: str, monthly_spending: str) -> dict:
    """Store financial information"""
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"💰 Storing financial info: {annual_income}, {monthly_spending}")
        
        agent.current_collected_data.update({
            "annual_income": annual_income,
            "monthly_spending": monthly_spending
        })
        
        print(f"✅ Financial info stored successfully")
        return {"success": True, "message": "Financial information saved successfully."}
        
    except Exception as e:
        print(f"❌ Error in store_financial_info_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def store_holiday_preferences_tool(holiday_preferences: str) -> dict:
    """Store holiday preferences"""
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"🏖️ Storing holiday preferences: {holiday_preferences}")
        
        agent.current_collected_data["holiday_preferences"] = holiday_preferences
        
        print(f"✅ Holiday preferences stored successfully")
        return {"success": True, "message": "Holiday preferences saved successfully."}
        
    except Exception as e:
        print(f"❌ Error in store_holiday_preferences_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def send_group_invites_tool(email_list: str) -> dict:
    """Send group invites to a list of email addresses"""
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"📧 Sending group invites to: {email_list}")
        
        # Parse email list
        emails = [email.strip() for email in email_list.replace(';', ',').split(',') if email.strip()]
        
        if not emails:
            return {"success": False, "error": "No valid email addresses provided"}
        
        # Get user's group code
        group_code = agent.current_collected_data.get("group_code")
        if not group_code:
            return {"success": False, "error": "No group code available"}
        
        # Send invites
        invite_results = []
        for email in emails:
            try:
                result = _run_async_safely(agent.email_service.send_group_invite(email, group_code, agent.current_user_id))
                invite_results.append(f"{email}: {'Success' if result.get('success') else 'Failed'}")
            except Exception as e:
                invite_results.append(f"{email}: Failed ({str(e)})")
        
        # Mark group invites as sent
        agent.current_collected_data["group_invites_sent"] = True
        agent.current_collected_data["invited_emails"] = emails
        
        # Check if onboarding is now complete
        is_onboarding_complete = agent._check_data_completeness()
        
        print(f"✅ Group invites sent: {invite_results}")
        print(f"🏁 Onboarding complete after group invites: {is_onboarding_complete}")
        
        if is_onboarding_complete:
            return {
                "success": True, 
                "message": f"Group invites sent to {len(emails)} email(s). Thank you, this completes your onboarding to Arny!",
                "results": invite_results,
                "onboarding_complete": True
            }
        else:
            return {
                "success": True, 
                "message": f"Group invites sent to {len(emails)} email(s)",
                "results": invite_results
            }
        
    except Exception as e:
        print(f"❌ Error in send_group_invites_tool: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@function_tool
def decline_group_invites_tool() -> dict:
    """Decline to send group invites"""
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"❌ User declined to send group invites")
        
        # Mark group invites as declined
        agent.current_collected_data["group_invites_declined"] = True
        
        # Check if onboarding is now complete
        is_onboarding_complete = agent._check_data_completeness()
        
        print(f"✅ Group invites declined")
        print(f"🏁 Onboarding complete after declining invites: {is_onboarding_complete}")
        
        if is_onboarding_complete:
            return {
                "success": True, 
                "message": "Group invites declined. You can always send invites later from the app. Thank you, this completes your onboarding to Arny!",
                "onboarding_complete": True
            }
        else:
            return {
                "success": True, 
                "message": "Group invites declined. You can always send invites later from the app."
            }
        
    except Exception as e:
        print(f"❌ Error in decline_group_invites_tool: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool
def validate_group_code_tool(group_code: str) -> dict:
    """Validate if a group code exists and join the group if it does"""
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"🔍 Validating group code: {group_code}")
        
        # Check if group exists
        group_exists = _run_async_safely(agent.db.check_group_exists(group_code))
        
        if group_exists:
            print(f"✅ Group code {group_code} exists, joining group")
            
            # Join the group as a member
            join_result = _run_async_safely(agent.db.join_group(agent.current_user_id, group_code))
            
            if join_result:
                agent.current_collected_data["group_code"] = group_code
                agent.current_collected_data["group_role"] = "member"
                print(f"✅ Successfully joined group {group_code} as member")
                return {
                    "success": True,
                    "group_exists": True,
                    "joined": True,
                    "message": f"Great! You've successfully joined the group with code {group_code}."
                }
            else:
                return {
                    "success": False,
                    "group_exists": True,
                    "joined": False,
                    "error": "Failed to join the group"
                }
        else:
            print(f"❌ Group code {group_code} does not exist")
            return {
                "success": False,
                "group_exists": False,
                "joined": False,
                "message": f"The group code {group_code} doesn't exist. Please check the code and try again, or type 'skip' to skip joining a group for now."
            }
        
    except Exception as e:
        print(f"❌ Error in validate_group_code_tool: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@function_tool
def skip_joining_group_tool() -> dict:
    """Skip joining group and create a personal group for the user"""
    try:
        agent = _get_onboarding_agent()
        if not agent:
            return {"error": "Agent not available"}
        
        print(f"⏭️ Skipping joining group for user {agent.current_user_id}")
        
        # Generate a unique group code for personal use
        personal_group_code = agent.group_generator.generate_group_code()
        
        # Try to create a personal group in database first
        try:
            personal_group_created = _run_async_safely(agent.db.create_group(
                agent.current_user_id, 
                personal_group_code, 
                group_name="Personal Group"
            ))
            
            if personal_group_created:
                print(f"✅ Personal group created in database with code: {personal_group_code}")
            else:
                print(f"⚠️ Failed to create personal group in database, but continuing with local setup")
        except Exception as db_error:
            print(f"⚠️ Database error creating group: {db_error}, but continuing with local setup")
        
        # Always set the group data locally regardless of database result
        agent.current_collected_data["group_code"] = personal_group_code
        agent.current_collected_data["group_role"] = "admin"
        agent.current_collected_data["group_joining_skipped"] = True
        
        print(f"✅ User set up as admin of personal group: {personal_group_code}")
        print(f"📊 Updated collected data: {agent.current_collected_data}")
        
        return {
            "success": True,
            "personal_group_code": personal_group_code,
            "message": "No problem! You can always invite family members later."
        }
        
    except Exception as e:
        print(f"❌ Error in skip_joining_group_tool: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# ==================== ENHANCED ONBOARDING AGENT CLASS ====================

class OnboardingAgent:
    """
    Enhanced LLM-driven onboarding agent with robust completion detection
    """
    
    def __init__(self):
        global _current_onboarding_agent
        
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.email_service = EmailService()
        self.db = DatabaseOperations()
        self.group_generator = GroupCodeGenerator()
        
        # Store this instance globally for tool access
        _current_onboarding_agent = self
        
        # Create the agent with enhanced tools using Agents SDK
        self.agent = Agent(
            name="Arny Onboarding Assistant",
            instructions=(
                "You are Arny AI, a helpful onboarding assistant for a travel planner "
                "personal assistant app. "
                "Your task is to obtain personal information from the "
                "app user as part of the onboarding process. Follow these steps:\n\n"
                "IMPORTANT: Continue conversations from where they left off based on collected data.\n\n"
                "1. JOINING A GROUP:\n"
                "   - When the user provides a Group Code, use validate_group_code_tool to check if it exists.\n"
                "   - If the group EXISTS: They will automatically join it as a member.\n"
                "   - If the group DOES NOT EXIST: Tell them the group doesn't exist and ask them to check the group code and try again, or type 'skip' to skip joining a group for now.\n"
                "   - If they say 'skip', 'no', 'none', or 'later', use skip_joining_group_tool ONLY ONCE.\n"
                "   - IMPORTANT: When a user skips joining a group, NEVER mention any specific group code to them. Just say that group joining has been skipped and they can join a group later.\n\n"
                "2. EMAIL SCANNING:\n"
                "   Ask about the user's Gmail or Outlook address, then use scan_email_for_profile_tool "
                "to fetch name, gender, birthdate, and city. IMPORTANT: If email scanning fails or returns an error, "
                "gracefully proceed to manual data collection without making the user feel like something went wrong. "
                "For example, if scanning fails, say: 'Email scanning currently does not work. I'll help you fill out your profile manually instead.' "
                "If you successfully extract ANY information (even partial), "
                "present what was found and ask the user to confirm or provide the missing details. "
                "For example: 'I found your name is John Smith, but I couldn't find your gender, birthdate, or city. "
                "Could you please provide these missing details?' Only if NO information is extracted at all should you "
                "ask for all details manually.\n\n"
                "3. PERSONAL INFORMATION:\n"
                "   Collect: Name, Gender, Birthdate (DD-MM-YYYY), Current City. Use store_personal_info_tool.\n\n"
                "4. JOB DETAILS:\n"
                "   Collect: Current Employer, Working Schedule, Holiday Frequency. Use store_job_details_tool.\n\n"
                "5. FINANCIAL INFORMATION:\n"
                "   Collect: Annual Income, Monthly Spending Budget. Use store_financial_info_tool.\n\n"
                "6. HOLIDAY PREFERENCES:\n"
                "   Collect: Preferred holiday types, destinations, activities. Use store_holiday_preferences_tool.\n\n"
                "7. GROUP INVITATIONS (ONLY if user has group_role = 'admin'):\n"
                "   - If the user joined an existing group under step '1. JOINING A GROUP' above (ie. group_role = 'member'), SKIP this step 7 entirely.\n"
                "   - If the user skipped joining a group under step '1. JOINING A GROUP' above (ie. group_role = 'admin'), ask: 'Would you like to invite people to join your new group? This can always be done later.' "
                "If the user says yes to inviting people to join a group, respond with 'Please invite users to your new group by providing their email addresses.' "
                "When they provide email addresses, respond with 'To confirm, I will be sending invites to {list all provided email addresses}. Are they correct?' "
                "If they confirm yes, use send_group_invites_tool with comma-separated email list to send the invitation emails. After successfully sending invites, check if this completes the onboarding process. "
                "If the user says no to inviting people to join a group, use decline_group_invites_tool to mark this step as complete. After declining invites, check if this completes the onboarding process. "
                "If email sending fails, gracefully explain that the group code can be shared manually.\n\n"
                "COMPLETION:\n"
                "Finally, ONLY when all the above onboarding process is completed, "
                "respond to the user and say: 'Thank you, this completes your onboarding to Arny!'\n\n"
                "IMPORTANT COMPLETION CHECK:\n"
                "After using send_group_invites_tool or decline_group_invites_tool, ALWAYS check if all onboarding steps are now complete. "
                "If they are complete (all data collected and group invites handled), immediately say: 'Thank you, this completes your onboarding to Arny!'\n\n"
                "IMPORTANT RULES:\n"
                "DO NOT call the same tool multiple times in one response. "
                "CONTINUE FROM WHERE THE CONVERSATION LEFT OFF - check collected data to see what step to proceed with. "
                "NEVER reveal specific group codes to users when they skip joining a group. "
                "ALWAYS use the appropriate store_*_tool when users provide information to ensure it gets saved. "
                "Handle email scanning failures gracefully without making the user feel like something is broken."
            ),
            model="o4-mini",
            tools=[
                scan_email_for_profile_tool,
                store_personal_info_tool,
                store_job_details_tool,
                store_financial_info_tool,
                store_holiday_preferences_tool,
                send_group_invites_tool,
                decline_group_invites_tool,
                validate_group_code_tool,
                skip_joining_group_tool
            ]
        )
    
    @openai_agents_sdk_retry
    async def _run_agent_with_retry(self, agent, input_data):
        """Run agent with retry logic applied"""
        return await Runner.run(agent, input_data)
    
    async def process_message(self, user_id: str, message: str, session_id: str, 
                            current_progress: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process message using OpenAI Agents SDK with enhanced completion detection
        """
        
        try:
            # Store user context for tool calls
            self.current_user_id = user_id
            self.current_collected_data = current_progress.get('collected_data', {})
            
            print(f"🔧 Processing message for user {user_id}: {message}")
            print(f"📊 Current collected data: {self.current_collected_data}")
            
            # Get conversation history
            conversation_history = current_progress.get('conversation_history', [])
            
            # Determine current step based on collected data
            current_step = self._determine_current_step_from_data(self.current_collected_data)
            print(f"🎯 Determined current step: {current_step}")
            
            # Create conversation context for the agent
            context_messages = []
            
            # Add system context about current progress
            if self.current_collected_data:
                progress_summary = self._create_progress_summary(self.current_collected_data, current_step)
                context_messages.append({
                    "role": "system", 
                    "content": f"CURRENT PROGRESS:\n{progress_summary}\n\nContinue from this point based on what's missing."
                })
            
            # Add conversation history
            for msg in conversation_history:
                context_messages.append(msg)
            
            # Process with agent using retry wrapper
            if not conversation_history:
                # First message in conversation
                print("🚀 Starting new conversation")
                result = await self._run_agent_with_retry(self.agent, message)
            else:
                # Continue conversation with context
                print(f"🔄 Continuing conversation with {len(context_messages)} previous messages")
                result = await self._run_agent_with_retry(self.agent, context_messages + [{"role": "user", "content": message}])
            
            # Extract response
            assistant_message = result.final_output
            print(f"🤖 Agent response: {assistant_message}")
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Keep conversation history manageable
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
            
            # ENHANCED: Check if onboarding is complete using multiple detection methods
            onboarding_complete = await self._detect_onboarding_completion_enhanced(assistant_message)
            
            # Update progress
            updated_progress = {
                'collected_data': self.current_collected_data,
                'conversation_history': conversation_history
            }
            
            print(f"📈 Updated collected data: {self.current_collected_data}")
            print(f"🏁 Onboarding complete: {onboarding_complete}")
            
            # Save progress to database
            if not onboarding_complete:
                current_step_enum = self._determine_current_step(self.current_collected_data)
                print(f"💾 Saving progress - Current step: {current_step_enum.value}")
                
                try:
                    await self.db.update_onboarding_progress(
                        user_id, 
                        current_step_enum, 
                        updated_progress
                    )
                    print(f"✅ Progress saved successfully")
                except Exception as db_error:
                    print(f"⚠️ Failed to save progress to database: {db_error}")
                    # Don't fail the entire process if database save fails
            
            return {
                'message': assistant_message,
                'onboarding_complete': onboarding_complete,
                'collected_data': self.current_collected_data,
                'progress_data': updated_progress
            }
            
        except Exception as e:
            print(f"❌ Error in process_message: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return a helpful error response
            return {
                'message': "I apologize, but I encountered an error. Could you please try again?",
                "onboarding_complete": False,
                "collected_data": self.current_collected_data,
                "error": str(e)
            }
        
    async def _detect_onboarding_completion_enhanced(self, message: str) -> bool:
        """Enhanced onboarding completion detection using multiple methods"""
        try:
            print(f"🔍 ENHANCED COMPLETION DETECTION START")
            print(f"📝 Analyzing message: '{message[:100]}...'")
            
            # Method 1: Data-based completion check (primary method)
            data_complete = self._check_data_completeness()
            print(f"📊 Data completeness result: {data_complete}")
            
            # Method 2: LLM-based completion check (secondary method)
            llm_complete = await self._detect_onboarding_completion_llm(message)
            print(f"🤖 LLM completion result: {llm_complete}")
            
            # Method 3: Phrase-based completion check (fallback method)
            phrase_complete = self._fallback_phrase_detection(message)
            print(f"📝 Phrase completion result: {phrase_complete}")
            
            # Completion logic: Data complete AND (LLM complete OR phrase complete)
            message_detection = llm_complete or phrase_complete
            final_complete = data_complete and message_detection
            
            print(f"🎯 COMPLETION LOGIC BREAKDOWN:")
            print(f"   • Data complete: {data_complete}")
            print(f"   • Message detection (LLM OR phrase): {message_detection}")
            print(f"   • LLM detected: {llm_complete}")
            print(f"   • Phrase detected: {phrase_complete}")
            print(f"   • Final result: {data_complete} AND {message_detection} = {final_complete}")
            
            return final_complete
            
        except Exception as e:
            print(f"❌ Error in enhanced completion detection: {e}")
            # Fallback to data completeness only
            fallback_result = self._check_data_completeness()
            print(f"🔄 Using fallback data completeness: {fallback_result}")
            return fallback_result
        
    def _check_data_completeness(self) -> bool:
        """Check if all required onboarding data has been collected"""
        try:
            required_fields = [
                "group_code",           # Group joining completed
                "email",               # Email provided
                "name", "gender", "birthdate", "city",  # Personal info
                "employer", "working_schedule", "holiday_frequency",  # Job details
                "annual_income", "monthly_spending",  # Financial info
                "holiday_preferences"  # Holiday preferences
            ]
            
            missing_fields = []
            for field in required_fields:
                if not self.current_collected_data.get(field):
                    missing_fields.append(field)
            
            is_complete = len(missing_fields) == 0
            
            # ENHANCED DEBUGGING: Print detailed status
            print(f"🔍 DETAILED DATA COMPLETENESS CHECK:")
            print(f"📋 All collected data: {self.current_collected_data}")
            print(f"❌ Missing required fields: {missing_fields}")
            print(f"✅ Basic data complete: {is_complete}")
            
            # Additional check for group invites (if applicable)
            group_role = self.current_collected_data.get("group_role")
            group_invites_sent = self.current_collected_data.get("group_invites_sent")
            group_invites_declined = self.current_collected_data.get("group_invites_declined")
            
            print(f"👥 Group role: {group_role}")
            print(f"📧 Group invites sent: {group_invites_sent}")
            print(f"❌ Group invites declined: {group_invites_declined}")
            
            if is_complete and group_role == "admin":
                # For admin users, check if they've been asked about group invites
                if not group_invites_sent and not group_invites_declined:
                    print(f"⏳ Admin user hasn't completed group invites step yet")
                    is_complete = False
                else:
                    print(f"✅ Admin user has completed group invites step")
            elif group_role == "member":
                print(f"✅ Member user - no group invites step required")
            else:
                print(f"❓ Unknown group role or no group role set")
            
            print(f"🏁 FINAL DATA COMPLETENESS RESULT: {is_complete}")
            return is_complete
            
        except Exception as e:
            print(f"❌ Error checking data completeness: {e}")
            return False
    
    @openai_responses_api_retry
    def _openai_responses_create_with_retry(self, **kwargs):
        """OpenAI Responses API call with retry logic"""
        return self.openai_client.responses.create(**kwargs)
    
    async def _detect_onboarding_completion_llm(self, message: str) -> bool:
        """LLM-based onboarding completion detection"""
        try:
            print(f"🔍 Using LLM to detect onboarding completion for message: '{message[:100]}...'")
            
            # Use OpenAI's o4-mini model to detect onboarding completion
            prompt = f"""You are analyzing a message from an AI onboarding assistant to determine if it indicates that the onboarding process has been completed.

Message to analyze: "{message}"

Determine if this message indicates that the onboarding process has been completed. Look for indicators such as:
- Explicit statements about completing onboarding (e.g., "this completes your onboarding", "onboarding is complete")
- Thank you messages that suggest finalization
- Statements that the user can now proceed to use the main features
- Congratulations or completion confirmations
- Messages indicating the user is ready to start using the app

Respond with only "YES" if the message clearly indicates onboarding completion, or "NO" if it does not."""

            response = self._openai_responses_create_with_retry(
                model="o4-mini",
                input=prompt
            )
            
            # Extract response
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                response_text = content_item.text.strip().upper()
                                
                                print(f"🤖 LLM completion detection result: '{response_text}'")
                                
                                is_complete = response_text == "YES"
                                if is_complete:
                                    print("🎉 LLM detected onboarding completion!")
                                else:
                                    print("🔄 LLM says onboarding not complete yet")
                                
                                return is_complete
            
            print(f"⚠️ LLM completion detection failed - no valid response")
            return False
            
        except Exception as e:
            print(f"❌ Error in LLM completion detection: {e}")
            return False
    
    def _fallback_phrase_detection(self, message: str) -> bool:
        """Fallback phrase-based completion detection"""
        try:
            completion_phrases = [
                "this completes your onboarding",
                "onboarding is complete",
                "welcome to arny ai",
                "you're now ready to start",
                "ready to start planning",
                "onboarding complete"
            ]
            
            message_lower = message.lower()
            for phrase in completion_phrases:
                if phrase in message_lower:
                    print(f"📝 Phrase detection found completion indicator: '{phrase}'")
                    return True
            
            print(f"📝 No completion phrases detected")
            return False
            
        except Exception as e:
            print(f"❌ Error in phrase completion detection: {e}")
            return False
    
    def _determine_current_step_from_data(self, collected_data: Dict[str, Any]) -> str:
        """Determine the current onboarding step from collected data (string version)"""
        if not collected_data.get("group_code"):
            return "group_code"
        elif not collected_data.get("email"):
            return "email_scan"
        elif not all([collected_data.get("name"), collected_data.get("gender"), 
                     collected_data.get("birthdate"), collected_data.get("city")]):
            return "personal_info"
        elif not all([collected_data.get("employer"), collected_data.get("working_schedule"), 
                     collected_data.get("holiday_frequency")]):
            return "job_details"
        elif not all([collected_data.get("annual_income"), collected_data.get("monthly_spending")]):
            return "financial_info"
        elif not collected_data.get("holiday_preferences"):
            return "holiday_preferences"
        elif (collected_data.get("group_role") == "admin" and 
              not collected_data.get("group_invites_sent") and 
              not collected_data.get("group_invites_declined")):
            return "group_invites"
        else:
            return "completed"
    
    def _create_progress_summary(self, collected_data: Dict[str, Any], current_step: str) -> str:
        """Create a summary of the current progress for the agent"""
        summary_parts = []
        
        # Group joining
        if collected_data.get("group_code"):
            role = collected_data.get("group_role", "member")
            summary_parts.append(f"✅ Group joining complete (Role: {role})")
        else:
            summary_parts.append("❌ Group joining pending")
        
        # Email
        if collected_data.get("email"):
            summary_parts.append(f"✅ Email: {collected_data['email']}")
        else:
            summary_parts.append("❌ Email pending")
        
        # Personal info
        personal_fields = ["name", "gender", "birthdate", "city"]
        personal_complete = all(collected_data.get(field) for field in personal_fields)
        if personal_complete:
            summary_parts.append("✅ Personal info complete")
        else:
            missing = [f for f in personal_fields if not collected_data.get(f)]
            summary_parts.append(f"❌ Personal info pending: {', '.join(missing)}")
        
        # Job details
        job_fields = ["employer", "working_schedule", "holiday_frequency"]
        job_complete = all(collected_data.get(field) for field in job_fields)
        if job_complete:
            summary_parts.append("✅ Job details complete")
        else:
            missing = [f for f in job_fields if not collected_data.get(f)]
            summary_parts.append(f"❌ Job details pending: {', '.join(missing)}")
        
        # Financial info
        financial_fields = ["annual_income", "monthly_spending"]
        financial_complete = all(collected_data.get(field) for field in financial_fields)
        if financial_complete:
            summary_parts.append("✅ Financial info complete")
        else:
            missing = [f for f in financial_fields if not collected_data.get(f)]
            summary_parts.append(f"❌ Financial info pending: {', '.join(missing)}")
        
        # Holiday preferences
        if collected_data.get("holiday_preferences"):
            summary_parts.append("✅ Holiday preferences complete")
        else:
            summary_parts.append("❌ Holiday preferences pending")
        
        # Group invites (for admin users)
        if collected_data.get("group_role") == "admin":
            if collected_data.get("group_invites_sent"):
                summary_parts.append("✅ Group invites sent")
            elif collected_data.get("group_invites_declined"):
                summary_parts.append("✅ Group invites declined")
            else:
                summary_parts.append("❌ Group invites pending")
        
        summary_parts.append(f"\nNEXT STEP: {current_step}")
        
        return "\n".join(summary_parts)
    
    def _determine_current_step(self, collected_data: Dict[str, Any]) -> OnboardingStep:
        """Determine the current onboarding step enum"""
        if not collected_data.get("group_code"):
            return OnboardingStep.GROUP_CODE
        elif not collected_data.get("email"):
            return OnboardingStep.EMAIL_SCAN
        elif not all([collected_data.get("name"), collected_data.get("gender"), 
                     collected_data.get("birthdate"), collected_data.get("city")]):
            return OnboardingStep.PERSONAL_INFO
        elif not all([collected_data.get("employer"), collected_data.get("working_schedule"), 
                     collected_data.get("holiday_frequency")]):
            return OnboardingStep.JOB_DETAILS
        elif not all([collected_data.get("annual_income"), collected_data.get("monthly_spending")]):
            return OnboardingStep.FINANCIAL_INFO
        elif not collected_data.get("holiday_preferences"):
            return OnboardingStep.HOLIDAY_PREFERENCES
        elif (collected_data.get("group_role") == "admin" and 
              not collected_data.get("group_invites_sent") and 
              not collected_data.get("group_invites_declined")):
            return OnboardingStep.GROUP_INVITES
        else:
            return OnboardingStep.COMPLETED
