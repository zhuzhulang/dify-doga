import logging

from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.__base.model_provider import ModelProvider

logger = logging.getLogger(__name__)

class DogaProvider(ModelProvider):
    def validate_provider_credentials(self, credentials):
        """
        Validate provider credentials

        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        # try:
        #     model_instance = self.get_model_instance(ModelType.LLM)

        #     # Use `doga` model for validate,
        #     model_instance.validate_credentials(model="doga", credentials=credentials)
        # except CredentialsValidateFailedError as ex:
        #     raise ex
        # except Exception as ex:
        #     logger.exception(f"{self.get_provider_schema().provider} credentials validate failed")
        #     raise ex
        pass