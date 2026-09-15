# FastAPI service and AWS deployment

The service has been validated locally. This document and the checked-in AWS
workflow are deployment references; the project does not maintain a public
cloud endpoint. AWS deployment is manual so publishing a Git tag cannot create
billable infrastructure.

## Local service

Install the API dependencies and point the service at a compatible checkpoint.
Use `.env.example` as the list of supported environment variables; the service
defaults to CPU inference:

```bash
pip install -r requirements-api.txt
export MODEL_CHECKPOINT=/absolute/path/to/best_checkpoint.pt
export MODEL_DEVICE=cpu
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/demo` for the interactive demo or
`http://localhost:8000/docs` for OpenAPI documentation.

Run the complete test suite with:

```bash
pip install -r requirements-test.txt
pytest -q
```

## Container

The checkpoint is intentionally excluded from the image:

```bash
docker build -t esm2-localization-api .
docker run --rm -p 8000:8000 \
  -e MODEL_CHECKPOINT=/app/model/best_checkpoint.pt \
  -v /absolute/checkpoint/folder:/app/model:ro \
  esm2-localization-api
```

Alternatively set `MODEL_S3_URI=s3://bucket/key`; the startup process downloads
the object once into `MODEL_CHECKPOINT` before starting the API.

## AWS ECS Fargate

1. Upload the private checkpoint to S3. Do not commit it or bake it into the
   public image.
2. Create an ECR repository, ECS cluster/service, Application Load Balancer,
   CloudWatch log group, task execution role, and task role.
3. Grant the task role `s3:GetObject` only for the checkpoint object.
4. Replace the placeholders in
   `deployment/aws/ecs-task-definition.template.json`, register it, and create
   the ECS service on port 8000.
5. Create a GitHub OIDC deployment role restricted to this repository and the
   production environment.
6. Configure GitHub repository variables: `AWS_REGION`, `AWS_DEPLOY_ROLE_ARN`,
   `ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE`, and `ECS_TASK_FAMILY`.
7. Run the `Deploy to AWS ECS` workflow manually only if a future deployment is
   intentionally required.

The workflow builds an immutable image tagged with the Git commit SHA, pushes
it to ECR, renders a new task-definition revision, and waits for ECS service
stability. The checked-in task definition is a bootstrap template; subsequent
deployments download the current registered definition from ECS.

The default template is CPU-only with 4 vCPU and 16 GiB RAM. Benchmark latency
and cost before production use. If GPU latency is required, use an ECS EC2 GPU
capacity provider rather than Fargate, which changes the infrastructure design.
