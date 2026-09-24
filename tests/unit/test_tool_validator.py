# import pytest
#
# from enterprise_harness.gateway import (
#     ToolDefinition,
#     ToolValidationError,
#     ToolValidator,
# )
#
#
# def test_validate_valid_arguments():
#     tool = ToolDefinition(
#         name="query_order",
#         input_schema={
#             "type": "object",
#             "properties": {
#                 "order_id": {
#                     "type": "string",
#                 },
#             },
#             "required": ["order_id"],
#         },
#     )
#
#     validator = ToolValidator()
#
#     arguments = {
#         "order_id": "1001",
#     }
#
#     result = validator.validate(
#         tool,
#         arguments,
#     )
#
#     assert result == arguments
#
#
# def test_validate_missing_required_argument():
#     tool = ToolDefinition(
#         name="query_order",
#         input_schema={
#             "type": "object",
#             "properties": {
#                 "order_id": {
#                     "type": "string",
#                 },
#             },
#             "required": ["order_id"],
#         },
#     )
#
#     validator = ToolValidator()
#
#     with pytest.raises(ToolValidationError):
#         validator.validate(
#             tool,
#             {},
#         )
#
#
# def test_validate_invalid_argument_type():
#     tool = ToolDefinition(
#         name="query_order",
#         input_schema={
#             "type": "object",
#             "properties": {
#                 "order_id": {
#                     "type": "string",
#                 },
#             },
#             "required": ["order_id"],
#         },
#     )
#
#     validator = ToolValidator()
#
#     with pytest.raises(ToolValidationError):
#         validator.validate(
#             tool,
#             {
#                 "order_id": 1001,
#             },
#         )