#!/bin/bash
set -eo pipefail

# Paths below are relative to this folder, wherever the script is run from
cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="prod"
NAMESPACE="default"
VERSION="3.5.3"
REGISTRY="webagent/backend"
HOST="api.webagent.example.com"
INGRESS_NAMESPACE="ingress-nginx"

# Print usage information
function print_usage {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -e, --environment   prod or uat: sets WEBAGENT_ENV, so the pod uses config/<env>.yaml (default: prod)"
  echo "  -n, --namespace     Kubernetes namespace (default: default)"
  echo "  -v, --version       Image tag to deploy (default: 3.5.3)"
  echo "  -i, --ingress-ns    Namespace of the nginx ingress controller (default: ingress-nginx; also --ingress-namespace)"
  echo "  -r, --registry      Image repository the cluster pulls from (default: webagent/backend)"
  echo "  -H, --host          Public host name for the ingress (default: api.webagent.example.com)"
  echo "  -h, --help          Show this help message"
  echo ""
  echo "Environment variables required:"
  echo "  OPENAI_API_KEY      OpenAI API key"
  echo "  WEBAGENT_SECURITY_SECRET_KEY  Random secret, at least 32 characters"
  echo "  WEBAGENT_API_KEY              Key clients send in the X-API-Key header"
  echo "  GRAFANA_ADMIN_PASSWORD        Grafana admin password"
  echo ""
  echo "Optional environment variables:"
  echo "  TOGETHER_API_KEY    Together AI API key"
  echo "  TAVILY_API_KEY      Tavily web search API key"
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
    -r|--registry)
      REGISTRY="$2"
      shift
      shift
      ;;
    -H|--host)
      HOST="$2"
      shift
      shift
      ;;
    -i|--ingress-ns|--ingress-namespace)
      INGRESS_NAMESPACE="$2"
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

# "production" was the old default name for prod
if [ "$ENVIRONMENT" = "production" ]; then
  ENVIRONMENT="prod"
fi
case "$ENVIRONMENT" in
  prod|uat) ;;
  *)
    echo -e "${RED}Error: environment must be prod or uat (got: $ENVIRONMENT)${NC}"
    exit 1
    ;;
esac

# Values substituted into the manifest with sed; secrets are never substituted (see below)
for value in "$NAMESPACE" "$VERSION" "$REGISTRY" "$HOST"; do
  if [[ -z "$value" || ! "$value" =~ ^[A-Za-z0-9._:/-]+$ ]]; then
    echo -e "${RED}Error: namespace, version, registry and host must be non-empty and use only letters, digits and . _ : / -${NC}"
    exit 1
  fi
done

# Check required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
  echo -e "${RED}Error: OPENAI_API_KEY environment variable is required${NC}"
  print_usage
  exit 1
fi

# The app refuses to start in prod without these
for required in WEBAGENT_SECURITY_SECRET_KEY WEBAGENT_API_KEY GRAFANA_ADMIN_PASSWORD; do
  if [ -z "${!required}" ]; then
    echo -e "${RED}Error: $required environment variable is required${NC}"
    print_usage
    exit 1
  fi
done

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
  # ServiceMonitor and PrometheusRule exist only with the Prometheus Operator
  if kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
    kubectl apply -f monitoring/prometheus-operator.yaml
  fi
  
  echo -e "${YELLOW}Deploying Grafana...${NC}"
  # The admin password comes from the environment; it is never stored in the repo or a file.
  kubectl create secret generic grafana-credentials -n monitoring \
    --from-literal=admin-user=admin \
    --from-literal=admin-password="$GRAFANA_ADMIN_PASSWORD" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl apply -f monitoring/grafana.yaml
  
  echo -e "${YELLOW}Configuring Grafana datasource...${NC}"
  kubectl apply -f monitoring/grafana-datasource.yaml
else
  echo -e "${GREEN}Monitoring namespace already exists${NC}"
fi

# Secrets go straight to the cluster: never written to a file or put through sed
# (a key containing "/" or "&" broke the old substitution).
echo -e "${YELLOW}Creating or updating secret webagent-secrets...${NC}"
kubectl create secret generic webagent-secrets -n "$NAMESPACE" \
  --from-literal=openai-api-key="$OPENAI_API_KEY" \
  --from-literal=together-api-key="${TOGETHER_API_KEY:-}" \
  --from-literal=security-secret-key="$WEBAGENT_SECURITY_SECRET_KEY" \
  --from-literal=api-key="$WEBAGENT_API_KEY" \
  --from-literal=tavily-api-key="${TAVILY_API_KEY:-}" \
  --from-literal=langsmith-api-key="${LANGSMITH_API_KEY:-}" \
  --dry-run=client -o yaml | kubectl apply -f -

# Process the deployment template (no secrets in it)
echo -e "${YELLOW}Generating deployment manifest...${NC}"
MANIFEST="$(mktemp)"
POLICY="$(mktemp)"
trap 'rm -f "$MANIFEST" "$POLICY"' EXIT
sed -e "s|^\( *image: \).*|\1$REGISTRY:$VERSION|" \
    -e "s|version: \"[^\"]*\"|version: \"$VERSION\"|g" \
    -e "/name: WEBAGENT_ENV/{n;s|value: \"prod\"|value: \"$ENVIRONMENT\"|;}" \
    -e "s|api.webagent.example.com|$HOST|g" \
    -e "s|namespace: default|namespace: $NAMESPACE|g" \
    deployment.yaml > "$MANIFEST"

# Apply the deployment
echo -e "${YELLOW}Deploying WebAgent backend ($ENVIRONMENT) to namespace $NAMESPACE...${NC}"
kubectl apply -f "$MANIFEST"

echo -e "${YELLOW}Applying NetworkPolicy (ingress from ${INGRESS_NAMESPACE} and monitoring)...${NC}"
sed -e "s|kubernetes.io/metadata.name: ingress-nginx|kubernetes.io/metadata.name: ${INGRESS_NAMESPACE}|" \
    -e "s|namespace: default|namespace: $NAMESPACE|g" \
    networkpolicy.yaml > "$POLICY"
kubectl apply -f "$POLICY"

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${YELLOW}Checking deployment status...${NC}"
kubectl rollout status deployment/webagent-backend -n "$NAMESPACE"

echo -e "${GREEN}WebAgent backend v$VERSION has been deployed to the $ENVIRONMENT environment in namespace $NAMESPACE${NC}"
echo -e "${YELLOW}To access Grafana dashboards, run:${NC}"
echo "kubectl port-forward -n monitoring svc/grafana 3000:3000"
echo -e "${YELLOW}Then open http://localhost:3000 in your browser${NC}" 