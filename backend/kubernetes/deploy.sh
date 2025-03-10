#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
NAMESPACE="default"
VERSION="2.4.0"

# Print usage information
function print_usage {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -e, --environment   Environment to deploy to (default: production)"
  echo "  -n, --namespace     Kubernetes namespace (default: default)"
  echo "  -v, --version       Version to deploy (default: 2.4.0)"
  echo "  -h, --help          Show this help message"
  echo ""
  echo "Environment variables required:"
  echo "  OPENAI_API_KEY      OpenAI API key"
  echo "  JWT_SECRET          Secret for JWT token generation"
  echo ""
  echo "Optional environment variables:"
  echo "  TOGETHER_API_KEY    Together AI API key"
  echo "  LANGSMITH_API_KEY   LangSmith API key for tracing"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -e|--environment)
      ENVIRONMENT="$2"
      shift
      shift
      ;;
    -n|--namespace)
      NAMESPACE="$2"
      shift
      shift
      ;;
    -v|--version)
      VERSION="$2"
      shift
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      print_usage
      exit 1
      ;;
  esac
done

# Check required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
  echo -e "${RED}Error: OPENAI_API_KEY environment variable is required${NC}"
  print_usage
  exit 1
fi

if [ -z "$JWT_SECRET" ]; then
  echo -e "${RED}Error: JWT_SECRET environment variable is required${NC}"
  print_usage
  exit 1
fi

# Create namespace if it doesn't exist
echo -e "${YELLOW}Checking if namespace $NAMESPACE exists...${NC}"
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
  echo -e "${YELLOW}Creating namespace $NAMESPACE...${NC}"
  kubectl create namespace $NAMESPACE
fi

# Deploy monitoring namespace and resources if they don't exist
echo -e "${YELLOW}Setting up monitoring resources...${NC}"
if ! kubectl get namespace monitoring &> /dev/null; then
  echo -e "${YELLOW}Creating monitoring namespace...${NC}"
  kubectl apply -f monitoring/namespace.yaml
  
  echo -e "${YELLOW}Deploying Prometheus...${NC}"
  kubectl apply -f monitoring/prometheus.yaml
  
  echo -e "${YELLOW}Deploying Grafana...${NC}"
  kubectl apply -f monitoring/grafana.yaml
  
  echo -e "${YELLOW}Configuring Grafana datasource...${NC}"
  kubectl apply -f monitoring/grafana-datasource.yaml
else
  echo -e "${GREEN}Monitoring namespace already exists${NC}"
fi

# Process the deployment template
echo -e "${YELLOW}Generating deployment manifest...${NC}"
cat deployment.yaml | \
  sed "s/\${OPENAI_API_KEY}/$OPENAI_API_KEY/g" | \
  sed "s/\${TOGETHER_API_KEY}/${TOGETHER_API_KEY:-}/g" | \
  sed "s/\${JWT_SECRET}/$JWT_SECRET/g" | \
  sed "s/\${LANGSMITH_API_KEY}/${LANGSMITH_API_KEY:-}/g" | \
  sed "s/webagent\/backend:2.4.0/webagent\/backend:$VERSION/g" | \
  sed "s/namespace: default/namespace: $NAMESPACE/g" > deployment-$ENVIRONMENT.yaml

# Apply the deployment
echo -e "${YELLOW}Deploying WebAgent backend to $ENVIRONMENT environment in namespace $NAMESPACE...${NC}"
kubectl apply -f deployment-$ENVIRONMENT.yaml

# Clean up
rm deployment-$ENVIRONMENT.yaml

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${YELLOW}Checking deployment status...${NC}"
kubectl rollout status deployment/webagent-backend -n $NAMESPACE

echo -e "${GREEN}WebAgent backend v$VERSION has been deployed to the $ENVIRONMENT environment in namespace $NAMESPACE${NC}"
echo -e "${YELLOW}To access Grafana dashboards, run:${NC}"
echo "kubectl port-forward -n monitoring svc/grafana 3000:3000"
echo -e "${YELLOW}Then open http://localhost:3000 in your browser${NC}" 