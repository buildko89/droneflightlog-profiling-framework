import os

import pandas as pd


class VideoOnlyInterpreter:
    """
    Builds an interpretation from video-only analysis artifacts.
    """

    def __init__(self, context, llm_client):
        self.context = context
        self.llm_client = llm_client

    def run_interpretation(self, output_file, mode="api"):
        prompt = self._create_prompt()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if mode == "export":
            workspace_dir = getattr(self.context, "workspace_dir", "workspace")
            prompt_file = os.path.join(workspace_dir, "video_llm_prompt.txt")
            os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
            with open(prompt_file, "w", encoding="utf-8") as file:
                file.write(prompt)
            text = (
                "# 動画単体解析 AI診断プロンプト出力\n\n"
                f"- **想定モデル**: {self.llm_client.model_name}\n"
                f"- **プロンプトファイル**: `{prompt_file}`\n\n"
                "API は呼び出していません。上記プロンプトを任意の LLM に渡してください。\n"
            )
        else:
            text = self.llm_client.generate_text(prompt)
            text = (
                "# 動画単体解析 AI診断\n\n"
                f"- **使用モデル**: {self.llm_client.model_name}\n\n"
                f"{text}"
            )

        self.context.set_data("video_llm_diagnosis", pd.DataFrame({"interpretation": [text]}))
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(text)
        self.context.add_log(f"Video-only interpretation saved to: {output_file}")
        return output_file

    def _create_prompt(self):
        parse_report = self.context.get_artifact("video_parse_report", {})
        feature_summary = self.context.get_artifact("video_feature_summary", {})
        events = self.context.get_data("video_events")
        features = self.context.get_data("video_features")

        prompt = """
あなたはドローン動画レビューの補助者です。
以下は動画単体解析の結果です。フライトログ、センサー値、PCA異常、GPS、IMU、バッテリー情報はありません。

重要な前提:
- ログが無いため、機体故障やセンサー異常を断定しないでください。
- 外部カメラの場合、画面の揺れを機体の揺れと断定しないでください。
- ローカルCVイベントは粗いヒューリスティック候補です。
- 観察できる変化、ブレ、視認性低下、急な動きの候補時刻を整理してください。

### 動画メタデータ
"""
        prompt += f"{parse_report}\n\n"
        prompt += "### 特徴量サマリー\n"
        prompt += f"{feature_summary}\n\n"

        if events is not None and not events.empty:
            prompt += "### 動画イベント候補\n"
            for row in events.head(100).to_dict("records"):
                prompt += f"- {row}\n"
            prompt += "\n"

        if features is not None and not features.empty:
            prompt += "### 動画特徴量サンプル\n"
            for row in features.head(30).to_dict("records"):
                prompt += f"- {row}\n"
            prompt += "\n"

        prompt += """
### 依頼内容
- 動画内で目立つ変化を時刻つきで整理してください。
- ブレ、露出、視認性、急な動きの候補を説明してください。
- 外部カメラの場合は手ブレや撮影者操作の可能性を明記してください。
- ログが無いため、テレメトリ上の異常とは言わないでください。
- 日本語で簡潔にまとめてください。
"""
        return prompt
