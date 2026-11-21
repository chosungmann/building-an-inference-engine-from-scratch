package book.badak.demo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.os.Bundle;
import android.util.Log;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import book.badak.Badak;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * 앱 번들에 든 모델로 번들에 든 사진을 분류해 화면에 보여 준다 (44장).
 *
 * <p>같은 결과를 "BADAK"으로 시작하는 줄로도 적는다(logcat). 맥에서 돌린
 * 결과와 글자 그대로 견주려는 것이다.
 */
public final class MainActivity extends Activity {
    private static final String TAG = "badak";

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        column.setPadding(48, 48, 48, 48);
        TextView title = new TextView(this);
        title.setText("badak " + Badak.version());
        title.setTextSize(20);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        column.addView(title);
        ImageView photo = new ImageView(this);
        try (InputStream stream = getAssets().open("cup.jpg")) {
            photo.setImageBitmap(BitmapFactory.decodeStream(stream));
        } catch (IOException error) {
            Log.e(TAG, "BADAK error " + error);
        }
        photo.setAdjustViewBounds(true);
        photo.setMaxHeight(900);
        photo.setPadding(0, 32, 0, 32);
        column.addView(photo);
        TextView result = new TextView(this);
        result.setTextSize(16);
        result.setTypeface(Typeface.MONOSPACE);
        column.addView(result);
        // 상태 표시줄과 제목 줄 밑으로 들어가지 않게 한 겹 감싼다.
        FrameLayout frame = new FrameLayout(this);
        frame.setFitsSystemWindows(true);
        frame.addView(column);
        setContentView(frame);

        TextView answer = new TextView(this);
        answer.setTextSize(14);
        answer.setPadding(0, 32, 0, 0);
        column.addView(answer);

        // 엔진은 화면을 그리는 스레드 밖에서 돌린다.
        new Thread(() -> {
            String text = classify();
            runOnUiThread(() -> result.setText(text));
            ask(answer);
        }).start();
    }

    private static final String QUESTION = "What is the capital of France?";
    /** SmolLM2가 학습할 때 쓴 기본 지시문. 모델의 대화 틀에 적혀 있다. */
    private static final String SYSTEM =
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

    /** 앱의 파일 폴더에 언어 모델이 있으면 질문 하나를 보낸다 (47장). */
    private void ask(TextView view) {
        File directory = new File(getFilesDir(), "smollm2_135m");
        if (!new File(directory, "model.safetensors").exists()) {
            Log.i(TAG, "BADAK llm 없음");
            return;
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (Badak.Llm llm = new Badak.Llm(directory.getPath())) {
            int written = llm.chat(SYSTEM, QUESTION, 32, piece -> {
                // 엔진이 조각마다 이 함수를 부른다. 엔진의 스레드에서 불리므로
                // 화면은 주 스레드에 맡긴다.
                bytes.write(piece, 0, piece.length);
                String soFar = new String(bytes.toByteArray(), StandardCharsets.UTF_8);
                runOnUiThread(() -> view.setText(QUESTION + "\n\n" + soFar));
                return true;
            });
            StringBuilder hex = new StringBuilder();
            for (byte value : bytes.toByteArray()) {
                hex.append(String.format(Locale.ROOT, "%02x", value));
            }
            Log.i(TAG, "BADAK llm " + written + " " + hex);
        } catch (RuntimeException error) {
            Log.e(TAG, "BADAK error " + error);
        }
        Log.i(TAG, "BADAK llm-done");
    }

    private static byte[] readAll(InputStream stream) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        byte[] piece = new byte[64 * 1024];
        for (int count = stream.read(piece); count >= 0; count = stream.read(piece)) {
            bytes.write(piece, 0, count);
        }
        return bytes.toByteArray();
    }

    private String classify() {
        StringBuilder text = new StringBuilder();
        // openFd는 번들 파일 안에서 이 자산이 놓인 자리를 알려 준다.
        // 자산이 눌려 있으면 실패한다.
        try (AssetFileDescriptor model = getAssets().openFd("mobilenetv2.tflite");
                Badak.Classifier classifier = new Badak.Classifier(model);
                InputStream photo = getAssets().open("cup.jpg");
                InputStream labels = getAssets().open("labels.txt")) {
            String[] names = new String(readAll(labels), StandardCharsets.UTF_8).split("\n");
            Log.i(TAG, "BADAK version " + Badak.version());
            Log.i(TAG, "BADAK cpu " + Badak.cpuFeatures());
            for (Badak.Score score : classifier.classify(readAll(photo), 5)) {
                int bits = Float.floatToRawIntBits(score.probability);
                Log.i(TAG, String.format(Locale.ROOT, "BADAK top %d %08x", score.index, bits));
                text.append(String.format(
                        Locale.ROOT, "%s  %.1f%%\n", names[score.index], 100 * score.probability));
            }
        } catch (IOException | RuntimeException error) {
            Log.e(TAG, "BADAK error " + error);
            text.append(error);
        }
        Log.i(TAG, "BADAK done");
        return text.toString();
    }
}
