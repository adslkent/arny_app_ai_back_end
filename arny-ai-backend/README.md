# Arny AI Backend

A comprehensive travel planning assistant backend built with Python, AWS Lambda, and Supabase. Features intelligent agents for flight and hotel search, personalized recommendations, and group travel coordination.

## Architecture

- **Onboarding Agent**: Handles user registration and profile setup
- **Supervisor Agent**: Routes requests and handles general conversation
- **Flight Agent**: Manages flight searches using Amadeus API
- **Hotel Agent**: Manages hotel searches using Amadeus API
- **Database**: PostgreSQL via Supabase with Row Level Security
- **Authentication**: Supabase Auth with JWT tokens
- **Deployment**: AWS Lambda with API Gateway

## Features

- ✅ User authentication (signup/signin)
- ✅ Interactive onboarding with AI-powered profile extraction
- ✅ Group/family travel coordination with group codes
- ✅ Flight search with Amadeus Flight Offers Search API
- ✅ Hotel search with Amadeus Hotel Search API
- ✅ Intelligent conversation routing
- ✅ Persistent chat history
- ✅ Email integration for profile scanning
- ✅ AWS Lambda serverless deployment

## Prerequisites

- Python 3.11+
- AWS CLI configured
- Node.js 18+ (for Serverless Framework)
- Supabase account
- OpenAI API key
- Amadeus for Developers account
- Google Cloud Console project (for Gmail integration)
- Microsoft Azure app registration (for Outlook integration)

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone <your-repo>
cd arny-ai-backend
pip install -r requirements.txt
```

### 2. Supabase Setup

1. **Create a Supabase project:**
   - Go to [supabase.com](https://supabase.com)
   - Click "New Project"
   - Choose organization and set project name: "arny-ai"
   - Set a secure database password
   - Wait for project to be ready

2. **Get Supabase credentials:**
   - Go to Settings → API
   - Copy the Project URL (SUPABASE_URL)
   - Copy the anon public key (SUPABASE_ANON_KEY)
   - Copy the service_role secret key (SUPABASE_SERVICE_ROLE_KEY)

3. **Set up database schema:**
   - Go to SQL Editor in Supabase dashboard
   - Copy and run the SQL from `database_schema.sql`
   - This creates all necessary tables, indexes, and RLS policies

4. **Configure authentication:**
   - Go to Authentication → Settings
   - Enable email confirmations if desired
   - Configure any additional auth providers

### 3. OpenAI Setup

1. **Get OpenAI API key:**
   - Go to [platform.openai.com](https://platform.openai.com)
   - Navigate to API keys
   - Create a new secret key
   - Copy the key (starts with sk-proj-)

### 4. Amadeus API Setup

1. **Create Amadeus for Developers account:**
   - Go to [developers.amadeus.com](https://developers.amadeus.com)
   - Create account and verify email
   - Create a new app in the dashboard

2. **Get API credentials:**
   - Copy API Key (AMADEUS_API_KEY)
   - Copy API Secret (AMADEUS_API_SECRET)
   - Use test.api.amadeus.com for testing (AMADEUS_BASE_URL)

### 5. Google OAuth Setup (for Gmail integration)

1. **Create Google Cloud project:**
   - Go to [console.cloud.google.com](https://console.cloud.google.com)
   - Create new project: "arny-ai-backend"
   - Enable Google People API and Gmail API

2. **Configure OAuth consent screen:**
   - Go to APIs & Services → OAuth consent screen
   - Choose "External" user type
   - Fill in app information
   - Add scopes: openid, email, profile, gmail.send

3. **Create credentials:**
   - Go to APIs & Services → Credentials
   - Create OAuth 2.0 Client IDs
   - Application type: Web application
   - Add authorized redirect URI: `http://localhost:8000/auth/google/callback`
   - Copy Client ID and Client Secret

### 6. Microsoft Azure Setup (for Outlook integration)

1. **Create Azure app registration:**
   - Go to [portal.azure.com](https://portal.azure.com)
   - Navigate to App registrations
   - Click "New registration"
   - Name: "Arny Travel Assistant"
   - Supported account types: "Accounts in any organizational directory and personal Microsoft accounts"
   - Redirect URI: Add `http://localhost:8000/auth/outlook/callback` as Web redirect URI

2. **Configure API permissions:**
   - Go to API permissions
   - Add Microsoft Graph permissions: User.Read, Mail.Send, Calendars.Read
   - Grant admin consent

3. **Get credentials:**
   - Copy Application (client) ID
   - Create client secret in Certificates & secrets

### 7. AWS Setup

1. **Install AWS CLI and configure:**
   ```bash
   aws configure
   # Enter your AWS Access Key ID
   # Enter your AWS Secret Access Key
   # Default region: us-east-1
   # Default output format: json
   ```

2. **Install Serverless Framework:**
   ```bash
   npm install -g serverless
   npm install serverless-python-requirements
   ```

### 8. Environment Configuration

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Fill in your credentials in .env:**
   ```bash
   # OpenAI
   OPENAI_API_KEY=sk-proj-your-openai-api-key-here

   # Supabase
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your-supabase-anon-key
   SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

   # Amadeus
   AMADEUS_API_KEY=your-amadeus-api-key
   AMADEUS_API_SECRET=your-amadeus-api-secret
   AMADEUS_BASE_URL=test.api.amadeus.com

   # Google OAuth
   GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your-google-client-secret
   GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback

   # Microsoft OAuth
   OUTLOOK_CLIENT_ID=your-outlook-client-id
   OUTLOOK_CLIENT_SECRET=your-outlook-client-secret
   OUTLOOK_REDIRECT_URI=http://localhost:8000/auth/outlook/callback
   ```

### 9. Deployment

1. **Deploy to AWS:**
   ```bash
   # Deploy to development stage
   serverless deploy --stage dev

   # Deploy to production stage
   serverless deploy --stage prod
   ```

2. **Set up environment variables in AWS:**
   - The serverless.yml file automatically sets environment variables from your .env
   - You can also set them manually in AWS Lambda console

### 10. Testing

1. **Test health endpoint:**
   ```bash
   curl https://your-api-id.execute-api.us-east-1.amazonaws.com/dev/health
   ```

2. **Test authentication:**
   ```bash
   curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/dev/auth/signup \
     -H "Content-Type: application/json" \
     -d '{"email": "test@example.com", "password": "password123"}'
   ```

## API Endpoints

### Authentication
- `POST /auth/signup` - User registration
- `POST /auth/signin` - User login
- `POST /auth/refresh` - Refresh session token
- `POST /auth/signout` - Sign out user

### Onboarding
- `POST /onboarding/chat` - Onboarding conversation
- `POST /onboarding/group/check` - Validate group code
- `POST /onboarding/group/create` - Create new group
- `POST /onboarding/group/join` - Join existing group

### Travel Chat
- `POST /chat` - Main travel conversation
- `POST /travel/chat` - Alternative travel chat endpoint

### User Management
- `GET|POST /user/status` - Get user status and onboarding completion

## Usage Examples

### 1. User Registration
```bash
curl -X POST https://your-api/dev/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword"
  }'
```

### 2. Onboarding Chat
```bash
curl -X POST https://your-api/dev/onboarding/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-uuid",
    "message": "ABC123",
    "access_token": "jwt-token"
  }'
```

### 3. Flight Search
```bash
curl -X POST https://your-api/dev/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-uuid",
    "message": "Find flights from Sydney to Los Angeles on March 15th",
    "access_token": "jwt-token"
  }'
```

### 4. Hotel Search
```bash
curl -X POST https://your-api/dev/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-uuid",
    "message": "Find hotels in New York from March 20 to March 25",
    "access_token": "jwt-token"
  }'
```

## Development

### Project Structure
```
arny-ai-backend/
├── src/
│   ├── agents/          # AI agents (onboarding, supervisor, flight, hotel)
│   ├── auth/            # Authentication services
│   ├── database/        # Database models and operations
│   ├── handlers/        # Lambda request handlers
│   ├── services/        # External API services (Amadeus, email)
│   ├── utils/           # Utilities (config, group codes)
│   └── main.py          # Lambda entry point
├── database_schema.sql  # PostgreSQL schema
├── requirements.txt     # Python dependencies
├── serverless.yml      # AWS deployment configuration
└── README.md           # This file
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
PYTHONPATH=/home/fazil/projects/arny/arny-ai-backend uv run pytest src/test/ -v

# Run specific test file
PYTHONPATH=/home/fazil/projects/arny/arny-ai-backend uv run pytest src/test/test_amadeus_service.py -k "asyncio or (not asyncio and not trio)" -v

# Local testing with serverless offline
serverless offline --stage dev
```

### Running Tests

The project uses pytest with asyncio support for testing. Tests are located in the `src/test/` directory.

**Available test files:**
- `test_amadeus_service.py` - Tests for Amadeus API service integration
- `test_amadeus_api.py` - Direct Amadeus API tests
- `test_env_vars.py` - Environment variable validation tests

**Running all tests:**
```bash
# Using uv (recommended)
PYTHONPATH=/home/fazil/projects/arny/arny-ai-backend uv run pytest src/test/ -v

# Using python directly (if pytest is installed globally)
PYTHONPATH=/home/fazil/projects/arny/arny-ai-backend python -m pytest src/test/ -v
```

**Running specific tests:**
```bash
# Run only Amadeus service tests
PYTHONPATH=/home/fazil/projects/arny/arny-ai-backend uv run pytest src/test/test_amadeus_service.py -v

# Run only asyncio tests (skip trio tests that require additional dependencies)
PYTHONPATH=/home/fazil/projects/arny/arny-ai-backend uv run pytest src/test/test_amadeus_service.py -k "asyncio or (not asyncio and not trio)" -v

# Run tests with short traceback
PYTHONPATH=/home/fazil/projects/arny/arny-ai-backend uv run pytest src/test/ -v --tb=short
```

**Test requirements:**
- Tests use `pytest` with `anyio` plugin for async support
- Mocking is done with `unittest.mock`
- Environment variables are mocked in tests
- No actual API calls are made during testing

## Monitoring and Logs

1. **CloudWatch Logs:**
   - Go to AWS CloudWatch
   - Check log groups: `/aws/lambda/arny-ai-backend-dev-*`

2. **Supabase Monitoring:**
   - Go to Supabase dashboard
   - Check Database → Logs for SQL queries
   - Check Auth → Users for user management

3. **API Gateway Monitoring:**
   - Go to AWS API Gateway
   - Check your API's monitoring tab for request metrics

## Security Considerations

- All API endpoints require valid JWT tokens
- Row Level Security (RLS) enabled on all Supabase tables
- User data is isolated per user/group
- Environment variables stored securely in AWS
- CORS properly configured for web frontends

## Troubleshooting

### Common Issues

1. **"Invalid or expired token" errors:**
   - Check if access_token is included in requests
   - Verify token hasn't expired (refresh if needed)
   - Ensure Supabase auth is properly configured

2. **"Configuration error" on deployment:**
   - Verify all environment variables are set
   - Check .env file has all required keys
   - Ensure API keys are valid and active

3. **Database connection errors:**
   - Verify Supabase URL and keys are correct
   - Check if database schema was created properly
   - Ensure RLS policies are enabled

4. **Amadeus API errors:**
   - Verify API keys are correct and active
   - Check if you have sufficient quota
   - Ensure you're using the test environment initially

5. **"Route not found" errors:**
   - Check API Gateway endpoint URLs
   - Verify the path matches exactly
   - Ensure function is deployed properly

### Getting Help

- Check CloudWatch logs for detailed error messages
- Verify environment variables in AWS Lambda console
- Test individual components using local Python scripts
- Check Supabase logs for database-related issues

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
