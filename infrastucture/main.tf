provider "aws" {
  region = var.aws_region
}

resource "aws_instance" "mlops_instance" {
  ami           = var.aws_ami
  instance_type = var.aws_instance_type

  tags = {
    Name = "MLOps-MovieRec-Instance"
  }

  # Add any additional configuration (security groups, key pairs, etc.)
}
